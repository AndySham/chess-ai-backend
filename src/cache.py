import torch
from torch import nn, Tensor
import re
import pandas as pd
import os
from tqdm.notebook import tqdm

class TestMetric():
    def __init__(self):
        pass

    def name(self) -> str:
        return "empty-test"

    def measure_model(self, model, loader) -> float:
        return float("nan")

class TrainingRegime():
    def __init__(self, parent_dir, no_runs=1):
        self.parent_dir = parent_dir
        self.no_runs = no_runs
        self.models = [None] * self.no_runs
        self.epoch_counts = [None] * self.no_runs
        self.results = [None] * self.no_runs
        self.tests: list[TestMetric] = []

    def regime_str(self) -> str:
        raise NotImplementedError

    # Regime -----------------------------------------------------------------------

    def loop_until(self, run_no, until_epoch):
        start_epoch = self.get_loaded_epoch_count(run_no) + 1
        for epoch_no in range(start_epoch, until_epoch + 1):
            self.train_until(run_no, epoch_no)
            self.test_and_append_results(run_no)
        self.save_model(run_no)
        self.save_results(run_no)

    # Training ---------------------------------------------------------------------

    def training_dataloader(self, run_no):
        raise NotImplementedError

    def training_step(self, run_no, model):
        def step(*loader_args):
            raise NotImplementedError
        return step
        
    def train_until(self, run_no, until_epoch):
        model = self.get_loaded_model(run_no)
        start_epoch = self.get_loaded_epoch_count(run_no) + 1
        step = self.training_step(run_no, model)
        loader = self.training_dataloader(run_no)

        model.train()

        for epoch_no in range(start_epoch, until_epoch + 1):
            for loader_args in tqdm(
                loader, 
                desc="[%s] Run #%s, Epoch #%s, Training" 
                % (self.regime_str(), run_no, epoch_no), 
                leave=False
            ):
                if type(loader_args) is tuple:
                    step(*loader_args)
                else:
                    step(loader_args)
                self.set_loaded_epoch_count(run_no, epoch_no)

    # Testing ----------------------------------------------------------------------

    def testing_dataloader(self, run_no):
        raise NotImplementedError

    def test(self, run_no):
        model = self.get_loaded_model(run_no)
        epoch_no = self.get_loaded_epoch_count(run_no)
        loader = self.testing_dataloader(run_no)

        measurements = [None] * len(self.tests)

        model.eval()

        for idx, test_metric in enumerate(self.tests):
            def it():
                return tqdm(
                    loader, 
                    desc="[%s] Run #%s, Epoch #%s, Testing [%s]" 
                    % (self.regime_str(), run_no, epoch_no, test_metric.name()), 
                    leave=False
                )
            measurements[idx] = test_metric.measure_model(model, it)
                
        return measurements

    def test_and_append_results(self, run_no):
        epoch_no = self.get_loaded_epoch_count(run_no)
        results = self.get_loaded_results(run_no)
        measurements = self.test(run_no)
        results["epoch"].append(epoch_no)
        for idx, test_metric in enumerate(self.tests):
            results[test_metric.name()].append(measurements[idx])
        self.set_loaded_results(run_no, results)

    # File Loading -----------------------------------------------------------------

    def get_files(self):
        filenames = os.listdir(self.parent_dir)
        models = [self.model_from_filename(fl) for fl in filenames]
        results = [self.results_from_filename(fl) for fl in filenames]
        return [*filter(lambda x: x, models)], [*filter(lambda x: x, results)]   

    def regime_filename_elems(self) -> list[str]:
        raise NotImplementedError

    def regime_prefix(self):
        return self.parent_dir + "-".join(self.regime_filename_elems())

    def run_prefix(self, run_no):
        return self.regime_prefix() + "-" + str(run_no) 

    def ensure_directory(self):
        os.makedirs(self.parent_dir, exist_ok=True)

    def cache(self, filename, f):
        self.ensure_directory()
        filepath = self.regime_prefix() + "-" + filename
        try:
            return torch.load(filepath)
        except FileNotFoundError:
            data = f()
            torch.save(data, filepath)
            return data

    # Epoch Counts -----------------------------------------------------------------

    def get_loaded_epoch_count(self, run_no):
        model = self.epoch_counts[run_no - 1]
        if model is None:
            model = self.epoch_counts[run_no - 1] = 0
        return model

    def set_loaded_epoch_count(self, run_no, epoch_count):
        self.epoch_counts[run_no - 1] = epoch_count

    # Model Snapshots --------------------------------------------------------------

    def model_filename(self, run_no, epoch_no):
        return self.run_prefix(run_no) + "-model-" + str(epoch_no) + ".pt" 

    def model_from_filename(self, filename):
        model_str = "^((?:[^-]+-)+)([0-9]+)-model-([0-9]+)\\.pt$"
        match = re.search(model_str, filename)
        if match == None:
            return None

        params = match.group(1)
        if params != "-".join(self.regime_filename_elems()) + "-":
            return None

        run_no = int(match.group(2))
        epoch_no = int(match.group(3))

        return (run_no, epoch_no)

    def new_model(self):
        raise NotImplementedError

    def get_loaded_model(self, run_no):
        model = self.models[run_no - 1]
        if model is None:
            model = self.models[run_no - 1] = self.new_model()
        return model

    def load_model(self, run_no, epoch_no):
        filename = self.model_filename(run_no, epoch_no)
        model = self.new_model()
        model.load_state_dict(torch.load(filename))
        return model

    def load_latest_models(self):
        model_filenames, _ = self.get_files()
        
        self.models = [None] * self.no_runs
        epochs_by_run = [None] * self.no_runs

        for run_no, epoch_no in model_filenames:
            curr = epochs_by_run[run_no - 1]
            if curr is None or curr < epoch_no:
                epochs_by_run[run_no - 1] = epoch_no

        for run_no in range(1, self.no_runs + 1):
            epoch_no = epochs_by_run[run_no - 1]
            if epoch_no is not None:
                self.models[run_no - 1] = self.load_model(run_no, epoch_no)
                self.epoch_counts[run_no - 1] = epoch_no

    def save_model(self, run_no):
        self.ensure_directory()
        model = self.get_loaded_model(run_no)
        epoch_no = self.get_loaded_epoch_count(run_no)
        filename = self.model_filename(run_no, epoch_no)
        torch.save(model.state_dict(), filename)

    # Metric Results ---------------------------------------------------------------

    def results_filename(self, run_no):
        return self.run_prefix(run_no) + "-results.csv"

    def results_from_filename(self, filename):
        results_str = "^((?:[^-]+-)+)([0-9]+)-results\\.csv$"
        match = re.search(results_str, filename)
        if match == None:
            return None

        params = match.group(1)
        if params != "-".join(self.regime_filename_elems()) + "-":
            return None

        run_no = int(match.group(2))

        return run_no

    def new_results(self):
        results = {}
        results["epoch"] = []
        for test_metric in self.tests:
            results[test_metric.name()] = []
        return results

    def load_results(self, run_no):
        filename = self.results_filename(run_no)
        df = pd.read_csv(filename)
        return df.to_dict(orient="list")

    def load_all_results(self):
        self.results = [None] * self.no_runs

        _, result_files = self.get_files()
        for run_no in result_files:
            self.results[run_no - 1] = self.load_results(run_no)

    def get_loaded_results(self, run_no):
        results = self.results[run_no - 1]
        if results is None:
            results = self.results[run_no - 1] = self.new_results()
        return results

    def set_loaded_results(self, run_no, results):
        self.results[run_no - 1] = results

    def save_results(self, run_no):
        self.ensure_directory()
        results = self.get_loaded_results(run_no)
        epochs = results["epoch"]
        results = results.copy()
        del results["epoch"]
        filename = self.results_filename(run_no)
        pd.DataFrame(results, index=epochs).to_csv(filename, index_label="epoch")

class TrainingCache():
    def __init__(self, model_dir, idstr):
        self.model_dir = model_dir
        self.idstr = idstr
        
    def model_from_filename(self, filename):
        model_str = "^%s-(.+)-r([0-9]+)-([^-]+)-model\\.pt" % self.idstr
        match = re.search(model_str, filename)
        if match == None:
            return None
        loss = match.group(1)
        run_no = int(match.group(2))
        epoch = self.epoch_from_str(match.group(3))

        return (loss, run_no, epoch)
    
    def results_from_filename(self, filename):
        results_str = "^%s-(.+)-r([0-9]+)-results\\.csv" % self.idstr
        match = re.search(results_str, filename)
        if match == None:
            return None
        loss = match.group(1)
        run_no = int(match.group(2))

        return (loss, run_no)
        
    def get_files(self):
        filenames = os.listdir(self.model_dir)
        models = [self.model_from_filename(fl) for fl in filenames]
        results = [self.results_from_filename(fl) for fl in filenames]
        return [*filter(lambda x: x, models)], [*filter(lambda x: x, results)]
        
    def load_all_results(self):
        results = {
            'vae': [],
            'iwae': [],
            'ciwae-05': [],
            'miwae-8-8': [],
            'piwae-8-8': [],
            'dreg-1-64': [],
        }
        
        _, result_files = self.get_files()
        for loss, run_no in result_files:
            data = self.load_results(loss, run_no)
            results[loss].append((run_no, data))
            
        return results
    
    #Loads the latest models into the test objects
    def load_latest_models(self, tests):
        model_filenames, _ = self.get_files()
        losses = [t.name for t in tests]
        
        for t in tests:
            loss = t.name
            _, run_no, epoch = max(filter(lambda x: x[0] == loss, model_filenames), key=lambda x: x[2])
            self.load_model(t.model, loss, run_no, epoch)
            t.finished_epochs = epoch
        
    def epoch_str(self, epoch = None):
        return "final" if epoch == None else "e" + str(epoch)
    
    def epoch_from_str(self, s):
        if s == "final":
            return None
        else:
            match = re.search("e([0-9]+)", s)
            if not match:
                raise ValueError
            return int(match.group(1))
            
    def start_prefix(self):
        return self.model_dir + self.idstr + "-start"
    
    def run_prefix(self, loss, run_no):
        return (
            self.model_dir 
            + self.idstr 
            + "-"
            + loss 
            + "-r"
            + str(run_no)
        )
        
    def start_model_str(self):
        return self.start_prefix() + "-model.pt"
        
    def model_str(self, epoch, *args, **kargs):
        return self.run_prefix(*args, **kargs) + "-" + self.epoch_str(epoch)  + "-model.pt"

    def results_str(self, *args, **kargs):
        return self.run_prefix(*args, **kargs) + "-results.csv"
        
    def load_start(self, model):
        return model.load_state_dict(torch.load(self.start_model_str()))
    
    def save_start(self, model):
        torch.save(model.state_dict(), self.start_model_str())
        
    def load_model(self, model, loss, run_no, epoch):
        filename = self.model_str(epoch, loss, run_no)
        return model.load_state_dict(torch.load(filename))
        
    def save_model(self, model, loss, run_no, epoch):
        filename = self.model_str(epoch, loss, run_no)
        torch.save(model.state_dict(), filename)
        
    def load_results(self, loss, run_no):
        filename = self.results_str(loss, run_no)
        return pd.read_csv(filename)
    
    def load_results_dict(self, loss, run_no):
        tmp = {k: list(i.values()) for (k, i) in mnist_train_data.load_results(loss, run_no).to_dict().items()}
        if 'Unnamed: 0' in tmp: del tmp['Unnamed: 0']
        if 'Unnamed: 0.1' in tmp: del tmp['Unnamed: 0.1']
        return tmp
        
    def save_results(self, test_metrics, loss, run_no):
        filename = self.results_str(loss, run_no)
        test_metrics.to_csv(filename)