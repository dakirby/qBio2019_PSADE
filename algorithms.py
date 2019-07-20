"""Contains the Algorithm class and subclasses as well as support classes and functions for running simulations"""


from distributed import as_completed
from subprocess import run
from subprocess import CalledProcessError, TimeoutExpired
from subprocess import STDOUT

from .data import Data
from .pset import PSet
from .pset import Trajectory

from .pset import NetModel, BNGLModel, SbmlModelNoTimeout
from .pset import OutOfBoundsException
from .pset import FailedSimulationError
from .printing import print0, print1, print2, PybnfError
from .objective import ObjectiveCalculator, ConstraintCounter

import logging
import numpy as np
import os
import re
import shutil
import copy
import sys
import traceback
import pickle
from glob import glob
from tornado import gen
from distributed.client import _wait
from concurrent.futures._base import CancelledError


logger = logging.getLogger(__name__)


class Result(object):
    """
    Container for the results of a single evaluation in the fitting algorithm
    """

    def __init__(self, paramset, simdata, name):
        """
        Instantiates a Result

        :param paramset: The parameters corresponding to this evaluation
        :type paramset: PSet
        :param simdata: The simulation results corresponding to this evaluation, as a nested dictionary structure.
        Top-level keys are model names and values are dictionaries whose keys are action suffixes and values are
        Data instances
        :type simdata: dict Returns a
        :param log: The stdout + stderr of the simulations
        :type log: list of str
        """
        self.pset = paramset
        self.simdata = simdata
        self.name = name
        self.score = None  # To be set later when the Result is scored.
        self.failed = False

    def normalize(self, settings):
        """
        Normalizes the Data object in this result, according to settings
        :param settings: Config value for 'normalization': a string representing the normalization type, a dict mapping
        exp files to normalization type, or None
        :return:
        """
        if settings is None:
            return

        for m in self.simdata:
            for suff in self.simdata[m]:
                if type(settings) == str:
                    self.simdata[m][suff].normalize(settings)
                elif suff in settings:
                    self.simdata[m][suff].normalize(settings[suff])

    def postprocess_data(self, settings):
        """
        Postprocess the Data objects in this result with a user-defined Python script
        :param settings: A dict that maps a tuple (model, suffix) to a Python filename to load.
        That file is expected to contain the definition for the function postprocess(data),
        which takes a Data object and returns a processed data object
        :return: None
        """
        for m, suff in settings:
            rawdata = self.simdata[m][suff]
            # This could generate all kinds of errors if the user's script is bad. Whatever happens, it's caught
            # by the caller of postprocess_data()
            # exec(settings[m][suff])
            # noinspection PyUnresolvedReferences
            # self.simdata[m][suff] = postprocess(rawdata)

            # Cleaner attempt - follows good practice and is probably faster, but makes it hard for the user to create
            # a new Data object if they want to do that.
            # However, they can do that by `dataclass = data.__class__` `newdata = dataclass()`
            # Import the user-specified script as a module
            import importlib.util
            spec = importlib.util.spec_from_file_location("postprocessor", settings[m, suff])
            postproc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(postproc)
            # Now postproc is the user-defined Python module

            self.simdata[m][suff] = postproc.postprocess(rawdata)

    def add_result(self, other):
        """
        Add simulation data of other models from another Result object into this Result object
        :param other: The other Result object
        :return:
        """
        self.simdata.update(other.simdata)


class FailedSimulation(Result):
    def __init__(self, paramset, name, fail_type, einfo=tuple([None, None, None])):
        """
        Instantiates a FailedSimulation

        :param paramset:
        :param log:
        :param name:
        :param fail_type: 0 - Exceeded walltime, 1 - Other crash
        :type fail_type: int
        :param einfo:
        :type einfo: tuple
        """
        super(FailedSimulation, self).__init__(paramset, None, name)
        self.fail_type = fail_type
        self.failed = True
        self.traceback = ''.join(traceback.format_exception(*einfo))

    def normalize(self, settings):
        return

    def postprocess_data(self, settings):
        return


def run_job(j, debug=False, failed_logs_dir=''):
    """
    Runs the Job j.
    This function is passed to Dask instead of j.run_simulation because if you pass j.run_simulation, Dask leaks memory
    associated with j.
    """
    try:
        return j.run_simulation(debug, failed_logs_dir)
    except RuntimeError as e:
        # Catch the error for running out of threads here - it's the only place outside dask where we can catch it.
        if e.args[0] == "can't start new thread":
            logger.error("Reached thread limit - can't start new thread")
            print0('Too many threads! See "Troubleshooting" in the documentation for how to deal with this problem')
            return FailedSimulation(j.params, j.job_id, 1)
        else:
            raise


class Job:
    """
    Container for information necessary to perform a single evaluation in the fitting algorithm
    """

    # Seeing these logs for cluster-based fitting requires configuring dask to log to the
    # "pybnf.algorithms.job" logger
    jlogger = logging.getLogger('pybnf.algorithms.job')

    def __init__(self, models, params, job_id, output_dir, timeout, calc_future, norm_settings, postproc_settings,
                 delete_folder=False):
        """
        Instantiates a Job

        :param models: The models to evaluate
        :type models: list of Model instances
        :param params: The parameter set with which to evaluate the model
        :type params: PSet
        :param job_id: Job identification; also the folder name that the job gets saved to
        :type job_id: str
        :param output_dir path to the directory where I should create my simulation folder
        :type output_dir: str
        :param calc_future: Future for an ObjectiveCalculator containing the objective function and experimental data,
        which we can use to calculate the objective value.
        :type calc_future: Future
        :param norm_settings: Config value for 'normalization': a string representing the normalization type, a dict
        mapping exp files to normalization type, or None
        :type norm_settings: Union[str, dict, NoneType]
        :param postproc_settings: dict mapping (model, suffix) tuples to the path of a Python postprocessing file to
        run on the result.
        :param delete_folder: If True, delete the folder and files created after the simulation runs
        :type delete_folder: bool
        """
        self.models = models
        self.params = params
        self.job_id = job_id
        self.calc_future = calc_future
        self.norm_settings = norm_settings
        self.postproc_settings = postproc_settings
        # Whether to show warnings about missing data if the job includes an objective evaluation. Toggle this after
        # construction if needed.
        self.show_warnings = False
        self.home_dir = os.getcwd()  # This is safe because it is called from the scheduler, not the workers.
        # Force absolute paths for bngcommand and output_dir, because workers do not get the relative path info.
        if output_dir[0] == '/':
            self.output_dir = output_dir
        else:
            self.output_dir = self.home_dir + '/' + output_dir
        self.timeout = timeout

        # Folder where we save the model files and outputs.
        self.folder = '%s/%s' % (self.output_dir, self.job_id)
        self.delete_folder = delete_folder

    def _name_with_id(self, model):
        return '%s_%s' % (model.name, self.job_id)

    def _run_models(self):
        ds = {}
        for model in self.models:
            model_file_prefix = self._name_with_id(model)
            model_with_params = model.copy_with_param_set(self.params)
            ds[model.name] = model_with_params.execute(self.folder, model_file_prefix, self.timeout)
        return ds

    def _copy_log_files(self, failed_logs_dir):
        if failed_logs_dir == '':
            self.jlogger.error('Cannot save log files without specified directory')
            return
        for m in self.models:
            lf = '%s/%s.log' % (self.folder, self._name_with_id(m))
            if os.path.isfile(lf):
                self.jlogger.debug('Copying log file %s' % lf)
                shutil.copy(lf, failed_logs_dir)

    def run_simulation(self, debug=False, failed_logs_dir=''):
        """Runs the simulation and reads in the result"""

        # Force absolute path for failed_logs_dir
        if len(failed_logs_dir) > 0 and failed_logs_dir[0] != '/':
            failed_logs_dir = self.home_dir + '/' + failed_logs_dir

        # The check here is in case dask decides to run the same job twice, both of them can complete.
        made_folder = False
        failures = 0
        while not made_folder:
            try:
                os.mkdir(self.folder)
                self.jlogger.debug('Created folder %s for simulation' % self.folder)
                made_folder = True
            except OSError:
                self.jlogger.warning('Failed to create folder %s, trying again.' % self.folder)
                failures += 1
                self.folder = '%s/%s_rerun%i' % (self.output_dir, self.job_id, failures)
                if failures > 1000:
                    self.jlogger.error('Job %s failed because it was unable to write to the Simulations folder' %
                                       self.job_id)
                    return FailedSimulation(self.params, self.job_id, 1)
        try:
            simdata = self._run_models()
            res = Result(self.params, simdata, self.job_id)
        except (CalledProcessError, FailedSimulationError):
            if debug:
                self._copy_log_files(failed_logs_dir)
            res = FailedSimulation(self.params, self.job_id, 1)
        except TimeoutExpired:
            if debug:
                self._copy_log_files(failed_logs_dir)
            res = FailedSimulation(self.params, self.job_id, 0)
        except FileNotFoundError:
            self.jlogger.exception('File not found during job %s. This should only happen if the fitting '
                                   'is already done.' % self.job_id)
            res = FailedSimulation(self.params, self.job_id, 2, sys.exc_info())
        except Exception:
            if debug:
                self._copy_log_files(failed_logs_dir)
            print1('A simulation failed with an unknown error. See the log for details, and consider reporting this '
                   'as a bug.')
            self.jlogger.exception('Unknown error during job %s' % self.job_id)
            res = FailedSimulation(self.params, self.job_id, 2, sys.exc_info())
        else:
            if self.calc_future is not None:
                res.normalize(self.norm_settings)
                try:
                    res.postprocess_data(self.postproc_settings)
                except Exception:
                    self.jlogger.exception('User-defined post-processing script failed')
                    traceback.print_exc()
                    print0('User-defined post-processing script failed')
                    res.score = np.inf
                else:
                    res.score = self.calc_future.result().evaluate_objective(res.simdata, show_warnings=self.show_warnings)
                    if res.score is None:
                        res.score = np.inf
                        logger.warning('Simulation corresponding to Result %s contained NaNs or Infs' % res.name)
                        logger.warning('Discarding Result %s as having an infinite objective function value' % res.name)
                res.simdata = None
        if self.delete_folder:
            if os.name == 'nt':  # Windows
                try:
                    shutil.rmtree(self.folder)
                    self.jlogger.debug('Removed folder %s' % self.folder)
                except OSError:
                    self.jlogger.error('Failed to remove folder %s.' % self.folder)
            else:
                try:
                    run(['rm', '-rf', self.folder], check=True, timeout=1800)
                    self.jlogger.debug('Removed folder %s' % self.folder)
                except (CalledProcessError, TimeoutExpired):
                    self.jlogger.error('Failed to remove folder %s.' % self.folder)

        return res


class JobGroup:
    """
    Represents a group of jobs that are identical replicates to be averaged together for smoothing
    """
    def __init__(self, job_id, subjob_ids):
        """
        :param job_id: The name of the Job this group is representing
        :param subjob_ids: A list of the ids of the identical replicate Jobs.
        """
        self.job_id = job_id
        self.subjob_ids = subjob_ids
        self.result_list = []
        self.failed = None

    def job_finished(self, res):
        """
        Called when one job in this group has finished
        :param res: Result object for the completed job
        :return: Boolean, whether everything in this job group has finished
        """
        # Handle edge cases of failed simulations - if we get one FailedSimulation, we declare the group is done,
        # and return a FailedSimulation object as the average
        if self.failed:
            # JobGroup already finished when a previous failed simulation came in.
            return False
        if isinstance(res, FailedSimulation):
            self.failed = res
            return True

        if res.name not in self.subjob_ids:
            raise ValueError('Job group %s received unwanted result %s' % (self.job_id, res.name))
        self.result_list.append(res)
        return len(self.result_list) == len(self.subjob_ids)

    def average_results(self):
        """
        To be called after all results are in for this group.
        Averages the results and returns a new Result object containing the averages

        :return: New Result object with the job_id of this JobGroup and the averaged Data as the simdata
        """
        if self.failed:
            self.failed.name = self.job_id
            return self.failed

        # Iterate through the models and suffixes in the simdata strucutre, and calculate the average for each
        # Data object it contains
        avedata = dict()
        for m in self.result_list[0].simdata:
            avedata[m] = dict()
            for suf in self.result_list[0].simdata[m]:
                avedata[m][suf] = Data.average([r.simdata[m][suf] for r in self.result_list])
        return Result(self.result_list[0].pset, avedata, self.job_id)


class MultimodelJobGroup(JobGroup):
    """
    A JobGroup to handle model-level parallelism
    """

    def average_results(self):
        """
        To be called after all results are in for this group.
        Combines all results from the submodels into a single Result object
        :return:
        """
        if self.failed:
            self.failed.name = self.job_id
            return self.failed

        # Merge all models into a single Result object
        final_result = Result(self.result_list[0].pset, dict(), self.job_id)
        for res in self.result_list:
            final_result.add_result(res)
        return final_result


class custom_as_completed(as_completed):
    """
    Subclass created to modify a section of dask.distributed code
    By using this subclass instead of as_completed, if you get an exception in a job,
    that exception is returned as the result, instead of the job disappearing.
    """
    @gen.coroutine
    def track_future(self, future):
        try:
            yield _wait(future)
        except CancelledError:
            pass
        if self.with_results:
            try:
                result = yield future._result(raiseit=True)
            except Exception as e:
                result = DaskError(e, traceback.format_exc())
        with self.lock:
            self.futures[future] -= 1
            if not self.futures[future]:
                del self.futures[future]
            if self.with_results:
                self.queue.put_nowait((future, result))
            else:
                self.queue.put_nowait(future)
            self._notify()


class DaskError:
    """
    Class representing the result of a job that failed due to a raised exception
    """
    def __init__(self, error, tb):
        self.error = error
        self.traceback = tb


class Algorithm(object):
    """
    A superclass containing the structures common to all metaheuristic and MCMC-based algorithms
    defined in this software suite
    """

    def __init__(self, config):
        """
        Instantiates an Algorithm with a Configuration object.  Also initializes a
        Trajectory instance to track the fitting progress, and performs various additional
        configuration that is consistent for all algorithms

        :param config: The fitting configuration
        :type config: Configuration
        """
        self.config = config
        self.exp_data = self.config.exp_data
        self.objective = self.config.obj
        logger.debug('Instantiating Trajectory object')
        self.trajectory = Trajectory(self.config.config['num_to_output'])
        self.job_id_counter = 0
        self.output_counter = 0
        self.job_group_dir = dict()
        self.fail_count = 0
        self.success_count = 0
        self.max_iterations = config.config['max_iterations']

        logger.debug('Creating output directory')
        if not os.path.isdir(self.config.config['output_dir']):
            os.mkdir(self.config.config['output_dir'])

        if self.config.config['simulation_dir']:
            self.sim_dir = self.config.config['simulation_dir'] + '/Simulations'
        else:
            self.sim_dir = self.config.config['output_dir'] + '/Simulations'
        self.res_dir = self.config.config['output_dir'] + '/Results'
        self.failed_logs_dir = self.config.config['output_dir'] + '/FailedSimLogs'

        # Generate a list of variable names
        self.variables = self.config.variables

        # Store a list of all Model objects. Change this as needed for compatibility with other parts
        logger.debug('Initializing models')
        self.model_list = self._initialize_models()

        self.bootstrap_number = None
        self.best_fit_obj = None
        self.calc_future = None  # Created during Algorithm.run()
        self.refine = False

    def reset(self, bootstrap):
        """
        Resets the Algorithm, keeping loaded variables and models

        :param bootstrap: The bootstrap number (None if not bootstrapping)
        :type bootstrap: int or None
        :return:
        """
        logger.info('Resetting Algorithm for another run')
        self.trajectory = Trajectory(self.config.config['num_to_output'])
        self.job_id_counter = 0
        self.output_counter = 0
        self.job_group_dir = dict()
        self.fail_count = 0
        self.success_count = 0

        if bootstrap is not None:
            self.bootstrap_number = bootstrap

            self.sim_dir = self.config.config['output_dir'] + '/Simulations-boot%s' % bootstrap
            self.res_dir = self.config.config['output_dir'] + '/Results-boot%s' % bootstrap
            self.failed_logs_dir = self.config.config['output_dir'] + '/FailedSimLogs-boot%s' % bootstrap
            for boot_dir in (self.sim_dir, self.res_dir, self.failed_logs_dir):
                if os.path.exists(boot_dir):
                    try:
                        shutil.rmtree(boot_dir)
                    except OSError:
                        logger.error('Failed to remove bootstrap directory '+boot_dir)
                os.mkdir(boot_dir)

        self.best_fit_obj = None

    @staticmethod
    def should_pickle(k):
        """
        Checks to see if key 'k' should be included in pickling.  Currently allows all entries in instance dictionary
        except for 'trajectory'

        :param k:
        :return:
        """
        return k not in set(['trajectory', 'calc_future'])

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if self.should_pickle(k)}

    def __setstate__(self, state):
        self.__dict__.update(state)
        try:
            backup_params = 'sorted_params_backup.txt' if not self.refine else 'sorted_params_refine_backup.txt'
            self.trajectory = Trajectory.load_trajectory('%s/%s' % (self.res_dir, backup_params),
                                                         self.config.variables, self.config.config['num_to_output'])
        except IOError:
            logger.exception('Failed to load trajectory from file')
            print1('Failed to load Results/sorted_params_backup.txt . Still resuming your run, but when I save the '
                   'best fits, it will only be the ones I\'ve seen since resuming.')
            self.trajectory = Trajectory(self.config.config['num_to_output'])

    def _initialize_models(self):
        """
        Checks initial BNGLModel instances from the Configuration object for models that
        can be reinstantiated as NetModel instances

        :return: list of Model instances
        """
        # Todo: Move to config or BNGL model class?
        home_dir = os.getcwd()
        os.chdir(self.config.config['output_dir'])  # requires creation of this directory prior to function call
        logger.debug('Copying list of models')
        init_model_list = copy.deepcopy(list(self.config.models.values()))  # keeps Configuration object unchanged
        final_model_list = []
        init_dir = os.getcwd() + '/Initialize'

        for m in init_model_list:
            if isinstance(m, BNGLModel) and m.generates_network:
                logger.debug('Model %s requires network generation' % m.name)

                if not os.path.isdir(init_dir):
                    logger.debug('Creating initialization directory: %s' % init_dir)
                    os.mkdir(init_dir)
                os.chdir(init_dir)

                gnm_name = '%s_gen_net' % m.name
                default_pset = PSet([var.set_value(var.default_value) for var in self.variables])
                m.save(gnm_name, gen_only=True, pset=default_pset)
                gn_cmd = [self.config.config['bng_command'], '%s.bngl' % gnm_name]
                if os.name == 'nt':  # Windows
                    # Explicitly call perl because the #! line in BNG2.pl is not supported.
                    gn_cmd = ['perl'] + gn_cmd
                try:
                    with open('%s.log' % gnm_name, 'w') as lf:
                        print2('Generating network for model %s.bngl' % gnm_name)
                        run(gn_cmd, check=True, stderr=STDOUT, stdout=lf, timeout=self.config.config['wall_time_gen'])
                except CalledProcessError as c:
                    logger.error("Command %s failed in directory %s" % (gn_cmd, os.getcwd()))
                    logger.error(c.stdout)
                    print0('Error: Initial network generation failed for model %s... see BioNetGen error log at '
                           '%s/%s.log' % (m.name, os.getcwd(), gnm_name))
                    exit(1)
                except TimeoutExpired:
                    logger.debug("Network generation exceeded %d seconds... exiting" %
                                  self.config.config['wall_time_gen'])
                    print0("Network generation took too long.  Increase 'wall_time_gen' configuration parameter")
                    exit(1)
                except:
                    tb = ''.join(traceback.format_list(traceback.extract_tb(sys.exc_info())))
                    logger.debug("Other exception occurred:\n%s" % tb)
                    print0("Unknown error occurred during network generation, see log... exiting")
                    exit(1)
                finally:
                    os.chdir(home_dir)

                logger.info('Output for network generation of model %s logged in %s/%s.log' %
                             (m.name, init_dir, gnm_name))
                final_model_list.append(NetModel(m.name, m.actions, m.suffixes, m.mutants, nf=init_dir + '/' + gnm_name + '.net'))
                final_model_list[-1].bng_command = m.bng_command
            else:
                logger.info('Model %s does not require network generation' % m.name)
                final_model_list.append(m)
        os.chdir(home_dir)
        return final_model_list

    def start_run(self):
        """
        Called by the scheduler at the start of a fitting run.
        Must return a list of PSets that the scheduler should run.

        Algorithm subclasses optionally may set the .name field of the PSet objects to give a meaningful unique
        identifier such as 'gen0ind42'. If so, they MUST BE UNIQUE, as this determines the folder name.
        Uniqueness will not be checked elsewhere.

        :return: list of PSets
        """
        raise NotImplementedError("Subclasses must implement start_run()")

    def got_result(self, res):
        """
        Called by the scheduler when a simulation is completed, with the pset that was run, and the resulting simulation
        data

        :param res: result from the completed simulation
        :type res: Result
        :return: List of PSet(s) to be run next or 'STOP' string.
        """
        raise NotImplementedError("Subclasses must implement got_result()")

    def add_to_trajectory(self, res):
        """
        Adds the information from a Result to the Trajectory instance
        """
        # Evaluate objective if it wasn't done on workers.
        if res.score is None:  # Check if the objective wasn't evaluated on the workers
            res.normalize(self.config.config['normalization'])
            # Do custom postprocessing, if any
            try:
                res.postprocess_data(self.config.postprocessing)
            except Exception:
                logger.exception('User-defined post-processing script failed')
                traceback.print_exc()
                print0('User-defined post-processing script failed')
                res.score = np.inf
            else:
                res.score = self.objective.evaluate_multiple(res.simdata, self.exp_data, self.config.constraints)
            if res.score is None:  # Check if the above evaluation failed
                res.score = np.inf
                logger.warning('Simulation corresponding to Result %s contained NaNs or Infs' % res.name)
                logger.warning('Discarding Result %s as having an infinite objective function value' % res.name)
                print1('Simulation data in Result %s has NaN or Inf values.  Discarding this parameter set' % res.name)
        logger.info('Adding Result %s to Trajectory with score %.4f' % (res.name, res.score))
        self.trajectory.add(res.pset, res.score, res.name)

    def random_pset(self):
        """
        Generates a random PSet based on the distributions and bounds for each parameter specified in the configuration

        :return:
        """
        logger.debug("Generating a randomly distributed PSet")
        pset_vars = []
        for var in self.variables:
            pset_vars.append(var.sample_value())
        return PSet(pset_vars)

    def random_latin_hypercube_psets(self, n):
        """
        Generates n random PSets with a latin hypercube distribution
        More specifically, the uniform_var and loguniform_var variables follow the latin hypercube distribution,
        while lognorm are randomized normally.

        :param n: Number of psets to generate
        :return:
        """
        logger.debug("Generating PSets using Latin hypercube sampling")
        num_uniform_vars = 0
        for var in self.variables:
            if var.type == 'uniform_var' or var.type == 'loguniform_var':
                num_uniform_vars += 1

        # Generate latin hypercube of dimension = number of uniformly distributed variables.
        rands = latin_hypercube(n, num_uniform_vars)
        psets = []

        for row in rands:
            # Initialize the variables
            # Convert the 0 to 1 random numbers to the required variable range
            pset_vars = []
            rowindex = 0
            for var in self.variables:
                if var.type == 'uniform_var':
                    rescaled_val = var.p1 + row[rowindex]*(var.p2-var.p1)
                    pset_vars.append(var.set_value(rescaled_val))
                    rowindex += 1
                elif var.type == 'loguniform_var':
                    rescaled_val = exp10(np.log10(var.p1) + row[rowindex]*(np.log10(var.p2)-np.log10(var.p1)))
                    pset_vars.append(var.set_value(rescaled_val))
                    rowindex += 1
                else:
                    pset_vars.append(var.sample_value())
            psets.append(PSet(pset_vars))
        return psets

    def make_job(self, params):
        """
        Creates a new Job using the specified params, and additional specifications that are already saved in the
        Algorithm object
        If smoothing is turned on, makes n identical Jobs and a JobGroup

        :param params:
        :type params: PSet
        :return: list of Jobs (of length equal to smoothing setting)
        """
        if params.name:
            job_id = params.name
        else:
            self.job_id_counter += 1
            job_id = 'sim_%i' % self.job_id_counter
        logger.debug('Creating Job %s' % job_id)
        if self.config.config['smoothing'] > 1:
            # Create multiple identical Jobs for use with smoothing
            newjobs = []
            newnames = []
            for i in range(self.config.config['smoothing']):
                thisname = '%s_rep%i' % (job_id, i)
                newnames.append(thisname)
                # calc_future is supposed to be None here - the workers don't have enough info to calculate the
                # objective on their own
                newjobs.append(Job(self.model_list, params, thisname,
                                   self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                                   self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files'])))
            new_group = JobGroup(job_id, newnames)
            for n in newnames:
                self.job_group_dir[n] = new_group
            return newjobs
        elif self.config.config['parallelize_models'] > 1:
            # Partition our model list into n different jobs
            newjobs = []
            newnames = []
            model_count = len(self.model_list)
            rep_count = self.config.config['parallelize_models']
            for i in range(rep_count):
                thisname = '%s_part%i' % (job_id, i)
                newnames.append(thisname)
                # calc_future is supposed to be None here - the workers don't have enough info to calculate the
                # objective on their own
                newjobs.append(Job(self.model_list[model_count*i//rep_count:model_count*(i+1)//rep_count],
                                   params, thisname, self.sim_dir, self.config.config['wall_time_sim'],
                                   self.calc_future, self.config.config['normalization'], dict(),
                                   bool(self.config.config['delete_old_files'])))
            new_group = MultimodelJobGroup(job_id, newnames)
            for n in newnames:
                self.job_group_dir[n] = new_group
            return newjobs
        else:
            # Create a single job
            return [Job(self.model_list, params, job_id,
                    self.sim_dir, self.config.config['wall_time_sim'], self.calc_future,
                    self.config.config['normalization'], self.config.postprocessing,
                    bool(self.config.config['delete_old_files']))]


    def output_results(self, name='', no_move=False):
        """
        Tells the Trajectory to output a log file now with the current best fits.

        This should be called periodically by each Algorithm subclass, and is called by the Algorithm class at the end
        of the simulation.
        :return:
        :param name: Custom string to add to the saved filename. If omitted, we just use a running counter of the
        number of times we've outputted.
        :param no_move: If True, overrides the config setting delete_old_files=2, and does not move the result to
        overwrite sorted_params.txt
        :type name: str
        """
        if name == '':
            name = str(self.output_counter)
            self.output_counter += 1
        if self.refine:
            name = 'refine_%s' % name
        filepath = '%s/sorted_params_%s.txt' % (self.res_dir, name)
        logger.info('Outputting results to file %s' % filepath)
        self.trajectory.write_to_file(filepath)

        # If the user has asked for fewer output files, each time we're here, move the new file to
        # Results/sorted_params.txt, overwriting the previous one.
        if self.config.config['delete_old_files'] >= 2 and not no_move:
            logger.debug("Overwriting previous 'sorted_params.txt'")
            noname_filepath = '%s/sorted_params.txt' % self.res_dir
            if os.path.isfile(noname_filepath):
                os.remove(noname_filepath)
            os.replace(filepath, noname_filepath)

    def backup(self, pending_psets=()):
        """
        Create a backup of this algorithm object that can be reloaded later to resume the run

        :param pending_psets: Iterable of PSets that are currently submitted as jobs, and will need to get re-submitted
        when resuming the algorithm
        :return:
        """

        logger.info('Saving a backup of the algorithm')
        # Save a backup of the PSets
        self.output_results(name='backup', no_move=True)

        # Pickle the algorithm
        # Save to a temporary file first, so we can't get interrupted and left with no backup.
        picklepath = '%s/alg_backup.bp' % self.config.config['output_dir']
        temppicklepath = '%s/alg_backup_temp.bp' % self.config.config['output_dir']
        try:
            f = open(temppicklepath, 'wb')
            pickle.dump((self, pending_psets), f)
            f.close()
            os.replace(temppicklepath, picklepath)
        except IOError as e:
            logger.exception('Failed to save backup of algorithm')
            print1('Failed to save backup of the algorithm.\nSee log for more information')
            if e.strerror == 'Too many open files':
                print0('Too many open files! See "Troubleshooting" in the documentation for how to deal with this '
                       'problem.')

    def get_backup_every(self):
        """
        Returns a number telling after how many individual simulation returns should we back up the algorithm.
        Makes a good guess, but could be overridden in a subclass
        """
        return self.config.config['backup_every'] * self.config.config['population_size'] * \
            self.config.config['smoothing']

    def add_iterations(self, n):
        """
        Adds n additional iterations to the algorithm.
        May be overridden in subclasses that don't use self.max_iterations to track the iteration count
        """
        self.max_iterations += n

    def run(self, client, resume=None, debug=False):
        """Main loop for executing the algorithm"""

        if self.refine:
            logger.debug('Setting up Simplex refinement of previous algorithm')

        backup_every = self.get_backup_every()
        sim_count = 0

        logger.debug('Generating initial parameter sets')
        if resume:
            psets = resume
            logger.debug('Resume algorithm with the following PSets: %s' % [p.name for p in resume])
        else:
            psets = self.start_run()

        if not os.path.isdir(self.failed_logs_dir):
            os.mkdir(self.failed_logs_dir)

        if self.config.config['local_objective_eval'] == 0 and self.config.config['smoothing'] == 1 and \
                self.config.config['parallelize_models'] == 1:
            calculator = ObjectiveCalculator(self.objective, self.exp_data, self.config.constraints)
            [self.calc_future] = client.scatter([calculator], broadcast=True)
        else:
            self.calc_future = None

        jobs = []
        pending = dict()  # Maps pending futures to tuple (PSet, job_id).
        for p in psets:
            jobs += self.make_job(p)
        jobs[0].show_warnings = True  # For only the first job submitted, show warnings if exp data is unused.
        logger.info('Submitting initial set of %d Jobs' % len(jobs))
        futures = []
        for job in jobs:
            f = client.submit(run_job, job, True, self.failed_logs_dir)
            futures.append(f)
            pending[f] = (job.params, job.job_id)
        pool = custom_as_completed(futures, with_results=True, raise_errors=False)
        backed_up = True
        while True:
            if sim_count % backup_every == 0 and not backed_up:
                self.backup(set([pending[fut][0] for fut in pending]))
                backed_up = True
            f, res = next(pool)
            if isinstance(res, DaskError):
                if isinstance(res.error, PybnfError):
                    raise res.error  # User-targeted error should be raised instead of skipped
                logger.error('Job failed with an exception')
                logger.error(res.traceback)
                res = FailedSimulation(pending[f][0], pending[f][1], 3)
            # Handle if this result is one of multiple instances for smoothing
            del pending[f]
            if self.config.config['smoothing'] > 1 or self.config.config['parallelize_models'] > 1:
                group = self.job_group_dir.pop(res.name)
                done = group.job_finished(res)
                if not done:
                    continue
                res = group.average_results()
            sim_count += 1
            backed_up = False
            if isinstance(res, FailedSimulation):
                if res.fail_type >= 1:
                    self.fail_count += 1
                tb = '\n'+res.traceback if res.fail_type == 1 else ''

                logger.debug('Job %s failed with code %d%s' % (res.name, res.fail_type, tb))
                if res.fail_type >= 1:
                    print1('Job %s failed' % res.name)
                else:
                    print1('Job %s timed out' % res.name)
                if self.success_count == 0 and self.fail_count >= 100:
                    raise PybnfError('Aborted because all jobs are failing',
                                     'Your simulations are failing to run. Logs from failed simulations are saved in '
                                     'the FailedSimLogs directory. For help troubleshooting this error, refer to '
                                     'https://pybnf.readthedocs.io/en/latest/troubleshooting.html#failed-simulations')
            else:
                self.success_count += 1
                logger.debug('Job %s complete' % res.name)

            self.add_to_trajectory(res)
            if res.score < self.config.config['min_objective']:
                logger.info('Minimum objective value achieved')
                print1('Minimum objective value achieved')
                break
            response = self.got_result(res)
            if response == 'STOP':
                self.best_fit_obj = self.trajectory.best_score()
                logger.info("Stop criterion satisfied with objective function value of %s" % self.best_fit_obj)
                print1("Stop criterion satisfied with objective function value of %s" % self.best_fit_obj)
                break
            else:
                new_futures = []
                for ps in response:
                    new_js = self.make_job(ps)
                    for new_j in new_js:
                        new_f = client.submit(run_job, new_j, (debug or self.fail_count < 10), self.failed_logs_dir)
                        pending[new_f] = (ps, new_j.job_id)
                        new_futures.append(new_f)
                logger.debug('Submitting %d new Jobs' % len(new_futures))
                pool.update(new_futures)

        logger.info("Cancelling %d pending jobs" % len(pending))
        client.cancel(list(pending.keys()))
        self.output_results('final')

        # Copy the best simulations into the results folder
        best_name = self.trajectory.best_fit_name()
        best_pset = self.trajectory.best_fit()
        logger.info('Copying simulation results from best fit parameter set to Results/ folder')
        for m in self.config.models:
            this_model = self.config.models[m]
            to_save = this_model.copy_with_param_set(best_pset)
            to_save.save_all('%s/%s_%s' % (self.res_dir, to_save.name, best_name))
            if self.config.config['delete_old_files'] == 0:
                for simtype, suf in this_model.suffixes:
                    if simtype == 'simulate':
                        ext = 'gdat'
                    else:  # parameter_scan
                        ext = 'scan'
                    if self.config.config['smoothing'] > 1:
                        best_name = best_name + '_rep0'  # Look for one specific replicate of the data
                    try:
                        shutil.copy('%s/%s/%s_%s_%s.%s' % (self.sim_dir, best_name, m, best_name, suf, ext),
                                    '%s' % self.res_dir)
                    except FileNotFoundError:
                        logger.error('Cannot find files corresponding to best fit parameter set')
                        print0('Could not find your best fit gdat file. This could happen if all of the simulations\n'
                               ' in your run failed, or if that gdat file was somehow deleted during the run.')
        if self.config.config['delete_old_files'] > 0 and self.config.config['save_best_data']:
            # Rerun the best fit parameter set so the gdat file(s) are saved in the Results folder.
            logger.info('Rerunning best fit parameter set to save data files.')
            # Enable saving files for SBML models
            for m in self.model_list:
                if isinstance(m, SbmlModelNoTimeout):
                    m.save_files = True
            finaljob = Job(self.model_list, best_pset, 'bestfit',
                           self.sim_dir, self.config.config['wall_time_sim'], None,
                           self.config.config['normalization'], self.config.postprocessing,
                           False)
            try:
                run_job(finaljob)
            except Exception:
                logger.exception('Failed to rerun best fit parameter set')
                print1('Failed to rerun best fit parameter set. See log for details')
            else:
                # Copy all gdat and scan to Results
                for fname in glob(self.sim_dir+'/bestfit/*.gdat') + glob(self.sim_dir+'/bestfit/*.scan'):
                    shutil.copy(fname, self.res_dir)
            # Disable saving files for SBML models (in case there is future bootstrapping or refinement)
            for m in self.model_list:
                if isinstance(m, SbmlModelNoTimeout):
                    m.save_files = False

        if self.bootstrap_number is None or self.bootstrap_number == self.config.config['bootstrap']:
            try:
                os.replace('%s/alg_backup.bp' % self.config.config['output_dir'],
                          '%s/alg_%s.bp' % (self.config.config['output_dir'],
                                            ('finished' if not self.refine else 'refine_finished')))
                logger.info('Renamed pickled algorithm backup to alg_%s.bp' %
                            ('finished' if not self.refine else 'refine_finished'))
            except OSError:
                logger.warning('Tried to move pickled algorithm, but it was not found')

        if (isinstance(self, SimplexAlgorithm) or self.config.config['refine'] != 1) and self.bootstrap_number is None:
            # End of fitting; delete unneeded files
            if self.config.config['delete_old_files'] >= 1:
                if os.name == 'nt':  # Windows
                    try:
                        shutil.rmtree(self.sim_dir)
                    except OSError:
                        logger.error('Failed to remove simulations directory '+self.sim_dir)
                else:
                    run(['rm', '-rf', self.sim_dir])  # More likely to succeed than rmtree()

        logger.info("Fitting complete")

    def cleanup(self):
        """
        Called before the program exits due to an exception.
        :return:
        """
        self.output_results('end')


class DifferentialEvolutionBase(Algorithm):

    def __init__(self, config):
        super(DifferentialEvolutionBase, self).__init__(config)

        self.mutation_rate = config.config['mutation_rate']
        self.mutation_factor = config.config['mutation_factor']
        self.max_iterations = config.config['max_iterations']
        self.stop_tolerance = config.config['stop_tolerance']

        self.strategy = config.config['de_strategy']
        options = ('rand1', 'rand2', 'best1', 'best2', 'all1', 'all2')
        if self.strategy not in options:
            raise PybnfError('Invalid differential evolution strategy "%s". Options are: %s' %
                             (self.strategy, ','.join(options)))

    def new_individual(self, individuals, base_index=None):
        """
        Create a new individual for the specified island, according to the set strategy

        :param base_index: The index to use for the new individual, or None for a random index.
        :return:
        """

        # Choose a starting parameter set (either a random one or the base_index specified)
        # and others to cross over (always random)

        if '1' in self.strategy:
            pickn = 3
        else:
            pickn = 5

        # Choose pickn random unique indices, or if base_index was given, choose base_index followed by pickn-1 unique
        # indices
        picks = np.random.choice(len(individuals), pickn, replace=False)
        if base_index is not None:
            if base_index in picks:
                # If we accidentally picked base_index, replace it with picks[0], preserving uniqueness in our list
                iswitch = list(picks).index(base_index)
                picks[iswitch] = picks[0]
            # Now overwrite picks[0] with base_index. If we have base_index, picks[0] was an "extra pick" we only needed
            # in case we sampled base_index and had to replace it.
            picks[0] = base_index
        base = individuals[picks[0]]
        others = [individuals[p] for p in picks[1:]]

        # Iterate through parameters; decide whether to mutate or leave the same.
        new_pset_vars = []
        for p in base:
            if np.random.random() < self.mutation_rate:
                if '1' in self.strategy:
                    update_val = self.mutation_factor * others[0].get_param(p.name).diff(others[1].get_param(p.name))
                else:
                    update_val = self.mutation_factor * others[0].get_param(p.name).diff(others[1].get_param(p.name)) +\
                                 self.mutation_factor * others[2].get_param(p.name).diff(others[3].get_param(p.name))
                new_pset_vars.append(p.add(update_val))
            else:
                new_pset_vars.append(p)

        return PSet(new_pset_vars)

    def start_run(self):
        return NotImplementedError("start_run() not implemented in DifferentialEvolutionBase class")

    def got_result(self, res):
        return NotImplementedError("got_result() not implemented in DifferentialEvolutionBase class")


class DifferentialEvolution(DifferentialEvolutionBase):
    """
    Implements the parallelized, island-based differential evolution algorithm
    described in Penas et al 2015.

    In some cases, I had to make my own decisions for specifics I couldn't find in the original paper. Namely:
    At each migration, a user-defined number of individuals are migrated from each island. For each individual, a
    random index is chosen; the same index for all islands. A random permutation is used to redistribute individuals
    with that index to different islands.

    Each island performs its migration individually, on the first callback when all islands are ready for that
    migration. It receives individuals from the migration iteration, regardless of what the current iteration is.
    This can sometimes lead to wasted effort.
    For example, suppose migration is set to occur at iteration 40, but island 1 has reached iteration 42 by the time
    all islands reach 40. Individual j on island 1 after iteration 42 gets replaced with individual j on island X
    after iteration 40. Some other island Y receives individual j on island 1 after iteration 40.

    """

    def __init__(self, config):
        """
        Initializes algorithm based on the config object.

        The following config keys specify algorithm parameters. For move information, see config_documentation.txt
        population_size
        num_islands
        max_iterations
        mutation_rate
        mutation_factor
        migrate_every
        num_to_migrate

        """
        super(DifferentialEvolution, self).__init__(config)

        self.num_islands = config.config['islands']
        self.num_per_island = int(config.config['population_size'] / self.num_islands)
        if config.config['population_size'] % config.config['islands'] != 0:
            logger.warning('Reduced population_size to %i to evenly distribute it over %i islands' %
                            (self.num_islands * self.num_per_island, self.num_islands))
        self.migrate_every = config.config['migrate_every']
        if self.num_islands == 1:
            self.migrate_every = np.inf
        self.num_to_migrate = config.config['num_to_migrate']

        self.island_map = dict()  # Maps each proposed PSet to its location (island, individual_i)
        self.iter_num = [0] * self.num_islands  # Count the number of completed iterations on each island
        self.waiting_count = []  # Count of the number of PSets that are pending evaluation on the current iteration of each island.
        self.individuals = []  # Nested list; individuals[i][j] gives individual j on island i.
        self.proposed_individuals = []  # Nested list of the same shape, gives individuals proposed for replacement in next generation
        self.fitnesses = []  # Nested list of same shape, gives fitness of each individual
        self.migration_ready = [0] * self.num_islands  # What migration number is each island ready for
        self.migration_done = [0] * self.num_islands  # What migration number has each island completed

        # These variables store data related to individual migrations.
        # Each one has migration number as keys. When the first island starts migration, the required entries are
        # created. When the last island completes migration, they are deleted to keep these structures small.
        self.migration_transit = dict()  # Store (PSet, fitness) tuples here that are getting migrated - one list per island
        self.migration_indices = dict()  # Which individual numbers are migrating in migration i - a single tuple for
        # each migration, used for all islands
        self.migration_perms = dict()  # How do we rearrange between islands on migration i?
        # For each migration, a list of num_to_migrate permutations of range(num_islands)

    def reset(self, bootstrap=None):
        super(DifferentialEvolution, self).reset(bootstrap)
        self.island_map = dict()
        self.iter_num = [0] * self.num_islands
        self.waiting_count = []
        self.individuals = []
        self.proposed_individuals = []
        self.fitnesses = []
        self.migration_ready = [0] * self.num_islands
        self.migration_done = [0] * self.num_islands

        self.migration_transit = dict()
        self.migration_indices = dict()
        self.migration_perms = dict()

    def start_run(self):
        if self.num_islands == 1:
            print2('Running Differential Evolution with population size %i for up to %i iterations' %
                   (self.num_per_island, self.max_iterations))
        else:
            print2('Running island-based Differential Evolution with %i islands of %i individuals each, '
                   'for up to %i iterations' % (self.num_islands, self.num_per_island, self.max_iterations))

        # Initialize random individuals
        if self.config.config['initialization'] == 'lh':
            psets = self.random_latin_hypercube_psets(self.num_islands*self.num_per_island)
            self.proposed_individuals = [psets[i * self.num_per_island: (i + 1) * self.num_per_island]
                                         for i in range(self.num_islands)]
        else:
            self.proposed_individuals = [[self.random_pset() for i in range(self.num_per_island)]
                                         for j in range(self.num_islands)]

        # Initialize the individual list to empty, will be filled with the proposed_individuals once their fitnesses
        # are computed.
        self.individuals = [[None
                             for i in range(self.num_per_island)]
                            for j in range(self.num_islands)]

        # Set all fitnesses to Inf, guaranteeing a replacement by the first proposed individual
        self.fitnesses = [[np.Inf
                           for i in range(self.num_per_island)]
                          for j in range(self.num_islands)]

        for i in range(len(self.proposed_individuals)):
            for j in range(len(self.proposed_individuals[i])):
                self.island_map[self.proposed_individuals[i][j]] = (i, j)
                if self.num_islands == 1:
                    self.proposed_individuals[i][j].name = 'gen0ind%i' % j
                else:
                    self.proposed_individuals[i][j].name = 'gen0isl%iind%i' % (i, j)

        self.waiting_count = [self.num_per_island] * self.num_islands

        return [ind for island in self.proposed_individuals for ind in island]

    def got_result(self, res):
        """
        Called when a simulation run finishes

        This is not thread safe - the Scheduler must ensure only one process at a time enters
        this function.
        (or, I should rewrite this function to make it thread safe)

        :param res: Result object
        :return:
        """

        pset = res.pset
        score = res.score

        # Calculate the fitness of this individual, and replace if it is better than the previous one.
        island, j = self.island_map.pop(pset)
        fitness = score
        if fitness <= self.fitnesses[island][j]:
            self.individuals[island][j] = pset
            self.fitnesses[island][j] = fitness

        self.waiting_count[island] -= 1

        # Determine if the current iteration is over for the current island
        if self.waiting_count[island] == 0:

            self.iter_num[island] += 1
            if min(self.iter_num) == self.iter_num[island]:
                # Last island to complete this iteration
                if self.iter_num[island] % self.config.config['output_every'] == 0:
                    self.output_results()
                if self.iter_num[island] % 10 == 0:
                    print1('Completed %i of %i iterations' % (self.iter_num[island], self.max_iterations))
                else:
                    print2('Completed %i of %i iterations' % (self.iter_num[island], self.max_iterations))
                print2('Current population fitnesses:')
                for l in self.fitnesses:
                    print2(sorted(l))

            if self.iter_num[island] == self.max_iterations:
                # Submit no more jobs for this island
                # Once all islands reach this, simulation is over.
                if min(self.iter_num) == self.max_iterations:
                    return 'STOP'
                else:
                    return []

            if self.iter_num[island] % self.migrate_every == 0:
                # This island prepares for migration
                migration_num = int(self.iter_num[island] / self.migrate_every)
                if max(self.migration_ready) < migration_num:
                    # This is the first island to reach this migration.
                    # Need to set global parameters for this migration.
                    self.migration_transit[migration_num] = [list() for i in range(self.num_islands)]
                    self.migration_indices[migration_num] = np.random.choice(range(self.num_per_island),
                                                                             size=self.num_to_migrate, replace=False)
                    self.migration_perms[migration_num] = [np.random.permutation(self.num_islands)
                                                           for i in range(self.num_to_migrate)]
                    logger.debug('Island %i just set up the migration.' % island)

                # Send the required PSets to migration_transit
                for j in self.migration_indices[migration_num]:
                    self.migration_transit[migration_num][island].append((self.individuals[island][j],
                                                                          self.fitnesses[island][j]))
                # Tell other islands that this one is ready for this migration.
                self.migration_ready[island] = migration_num

            if self.migration_done[island] < min(self.migration_ready):
                # This island performs a migration
                logger.debug('Island %i is migrating!' % island)
                migration_num = self.migration_done[island] + 1

                # Fetch the appropriate new individuals from migration_transit
                for migrater_index in range(self.num_to_migrate):
                    j = self.migration_indices[migration_num][migrater_index]  # Index of the individual
                    newisland = self.migration_perms[migration_num][migrater_index][island]
                    self.individuals[island][j], self.fitnesses[island][j] = \
                        self.migration_transit[migration_num][newisland][migrater_index]

                    logger.debug('Island %i gained new individual with fitness %f' % (island, self.fitnesses[island][j]))

                self.migration_done[island] = migration_num
                if min(self.migration_done) == migration_num:
                    # This is the last island to complete this migration
                    # Delete the migration data to free space.
                    del self.migration_transit[migration_num]
                    del self.migration_perms[migration_num]
                    del self.migration_indices[migration_num]

            # Set up the next generation
            best = np.argmin(self.fitnesses[island])
            for jj in range(self.num_per_island):
                if 'best' in self.strategy:
                    new_pset = self.new_individual(self.individuals[island], best)
                elif 'all' in self.strategy:
                    new_pset = self.new_individual(self.individuals[island], jj)
                else:
                    new_pset = self.new_individual(self.individuals[island])
                # If the new pset is a duplicate of one already in the island_map, it will cause problems.
                # As a workaround, perturb it slightly.
                while new_pset in self.island_map:
                    new_pset = PSet([v.add(np.random.uniform(-1e-6, 1e-6)) for v in new_pset])
                self.proposed_individuals[island][jj] = new_pset
                self.island_map[new_pset] = (island, jj)
                if self.num_islands == 1:
                    new_pset.name = 'gen%iind%i' % (self.iter_num[island], jj)
                else:
                    new_pset.name = 'gen%iisl%iind%i' % (self.iter_num[island], island, jj)

            self.waiting_count[island] = self.num_per_island

            if self.iter_num[island] % 20 == 0:
                logger.info('Island %i completed %i iterations' % (island, self.iter_num[island]))
                # print(sorted(self.fitnesses[island]))

            # Convergence check
            if (np.min(self.fitnesses) != 0) and (np.max(self.fitnesses) / np.min(self.fitnesses) < 1. + self.stop_tolerance):
                return 'STOP'

            # Return a copy, so our internal data structure is not tampered with.
            return copy.copy(self.proposed_individuals[island])

        else:
            # Add no new jobs, wait for this generation to complete.
            return []


class AsynchronousDifferentialEvolution(DifferentialEvolutionBase):
    """
    Implements a simple asynchronous differential evolution algorithm.

    Contains no islands or migrations. Instead, each time a PSet finishes, proposes a new PSet at the same index using
    the standard DE formula and whatever the current population happens to be at the time.

    """

    def __init__(self, config):
        """
        Initializes algorithm based on the config object.

        """
        super(AsynchronousDifferentialEvolution, self).__init__(config)

        self.population_size = config.config['population_size']

        self.sims_completed = 0
        self.individuals = []  # List of individuals
        self.fitnesses = []  # List of same shape, gives fitness of each individual

    def reset(self, bootstrap=None):
        super(AsynchronousDifferentialEvolution, self).reset(bootstrap)
        self.sims_completed = 0
        self.individuals = []
        self.fitnesses = []

    def start_run(self):
        print2('Running Asyncrhonous Differential Evolution with population size %i for up to %i iterations' %
               (self.population_size, self.max_iterations))

        # Initialize random individuals
        if self.config.config['initialization'] == 'lh':
            self.individuals = self.random_latin_hypercube_psets(self.population_size)
        else:
            self.individuals = [self.random_pset() for i in range(self.population_size)]

        # Set all fitnesses to Inf, guaranteeing a replacement by the first proposed individual.
        # The first replacement will replace with a copy of the same PSet, with the correct objective calculated.
        self.fitnesses = [np.Inf for i in range(self.population_size)]

        for i in range(len(self.individuals)):
            self.individuals[i].name = 'gen0ind%i' % i

        return copy.deepcopy(self.individuals)

    def got_result(self, res):
        """
        Called when a simulation run finishes

        :param res: Result object
        :return:
        """

        pset = res.pset
        fitness = res.score

        gen = int(re.search('(?<=gen)\d+', pset.name).group(0))
        j = int(re.search('(?<=ind)\d+', pset.name).group(0))

        if fitness <= self.fitnesses[j]:
            self.individuals[j] = pset
            self.fitnesses[j] = fitness

        self.sims_completed += 1

        # Do various "per iteration" stuff
        if self.sims_completed % self.population_size == 0:
            iters_complete = self.sims_completed / self.population_size
            if iters_complete % self.config.config['output_every'] == 0:
                self.output_results()
            if iters_complete % 10 == 0:
                print1('Completed %i of %i simulations' % (self.sims_completed, self.max_iterations * self.population_size))
            else:
                print2('Completed %i of %i simulations' % (self.sims_completed, self.max_iterations * self.population_size))
            print2('Current population fitnesses:')
            print2(sorted(self.fitnesses))
            if iters_complete % 20 == 0:
                logger.info('Completed %i simulations' % self.sims_completed)
            if iters_complete >= self.max_iterations:
                return 'STOP'
            # Convergence check
            if np.max(self.fitnesses) / np.min(self.fitnesses) < 1. + self.stop_tolerance:
                return 'STOP'

        if 'best' in self.strategy:
            best = np.argmin(self.fitnesses)
            new_pset = self.new_individual(self.individuals, best)
        elif 'all' in self.strategy:
            new_pset = self.new_individual(self.individuals, j)
        else:
            new_pset = self.new_individual(self.individuals)
        new_pset.name = 'gen%iind%i' % (gen+1, j)

        return [new_pset]


def latin_hypercube(nsamples, ndims):
    """
    Latin hypercube sampling.

    Returns a nsamples by ndims array, with entries in the range [0,1]
    You'll have to rescale them to your actual param ranges.
    """
    if ndims == 0:
        # Weird edge case - needed for other code counting on result having a number of rows
        return np.zeros((nsamples, 0))
    value_table = np.transpose(np.array([[i/nsamples + 1/nsamples * np.random.random() for i in range(nsamples)]
                                         for dim in range(ndims)]))
    for dim in range(ndims):
        np.random.shuffle(value_table[:, dim])
    return value_table


class ModelCheck(object):
    """
    An algorithm that just checks the fit quality for a job with no free parameters.

    Does not subclass Algorithm. To run, instead call run_check() with no Cluster.
    """

    def __init__(self, config):
        """
        Instantiates ModelCheck with a Configuration object.
        :param config: The fitting configuration
        :type config: Configuration
        """
        self.config = config
        self.exp_data = self.config.exp_data
        self.objective = self.config.obj
        self.bootstrap_number = None

        logger.debug('Creating output directory')
        if not os.path.isdir(self.config.config['output_dir']):
            os.mkdir(self.config.config['output_dir'])

        if self.config.config['simulation_dir']:
            self.sim_dir = self.config.config['simulation_dir'] + '/Simulations'
        else:
            self.sim_dir = self.config.config['output_dir'] + '/Simulations'

        # Store a list of all Model objects.
        self.model_list = copy.deepcopy(list(self.config.models.values()))

    def run_check(self, debug=False):
        """Main loop for executing the algorithm"""

        print1('Running model checking on the given model(s)')

        empty = PSet([])
        empty.name = 'check'
        job = Job(self.model_list, empty, 'check', self.sim_dir, self.config.config['wall_time_sim'], None,
                  None, dict(), delete_folder=False)
        result = run_job(job, debug, self.sim_dir)

        if isinstance(result, FailedSimulation):
            print0('Simulation failed.')
            return

        result.normalize(self.config.config['normalization'])
        try:
            result.postprocess_data(self.config.postprocessing)
        except Exception:
            logger.exception('User-defined post-processing script failed')
            traceback.print_exc()
            print0('User-defined post-processing script failed. Exiting')
            return

        result.score = self.objective.evaluate_multiple(result.simdata, self.exp_data, self.config.constraints)
        if result.score is None:
            print0('Simulation contained NaN or Inf values. Cannot calculate objective value.')
            return
        print0('Objective value is %s' % result.score)
        if len(self.config.constraints) > 0:
            counter = ConstraintCounter()
            fail_count = counter.evaluate_multiple(result.simdata, self.exp_data, self.config.constraints)
            total = sum([len(cset.constraints) for cset in self.config.constraints])
            print('Satisfied %i out of %i constraints' % (total-fail_count, total))
            for cset in self.config.constraints:
                cset.output_itemized_eval(result.simdata, self.sim_dir)

def exp10(n):
    """
    Raise 10 to the power of a possibly user-defined value, and raise a helpful error if it overflows
    :param n: A float
    :return: 10.** n
    """
    try:
        with np.errstate(over='raise'):
            ans = 10.**n
    except (OverflowError, FloatingPointError):
        logger.error('Overflow error in exp10()')
        logger.error(''.join(traceback.format_stack()))  # Log the entire traceback
        raise PybnfError('Overflow when calculating 10^%d\n'
                         'Logs are saved in bnf.log\n'
                         'This may be because you declared a lognormal_var or a logvar, and specified the '
                         'arguments in regular space instead of log10 space.' % n)
    return ans
