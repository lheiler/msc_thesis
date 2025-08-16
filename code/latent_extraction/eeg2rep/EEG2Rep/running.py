from torch.utils.data import DataLoader
# Import Project Modules ---------------------------------------------------------
from utils import dataset_class
from Models.model import Encoder_factory, count_parameters
from Models.loss import get_loss_module
from Models.utils import load_model
from trainer import *
# --------- For Logistic Regression--------------------------------------------------
from eval import fit_lr,  make_representation
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

####
logger = logging.getLogger('__main__')


def _build_fif_dataloaders(config):
    """
    Build DataLoaders from TUHFIF60sDataset provided by external gen_dataset.py.
    We create synthetic labels (zeros) for compatibility with existing API.
    """
    import importlib.util
    from pathlib import Path
    import torch
    import numpy as np

    fif_py = config.get('fif_dataset_py')
    fif_root = config.get('fif_root')
    if not fif_root:
        return None
    spec = importlib.util.spec_from_file_location("gen_dataset", fif_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    TUHFIF60sDataset = getattr(module, 'TUHFIF60sDataset')

    dataset = TUHFIF60sDataset(Path(fif_root), segment_len_sec=config.get('fif_segment_len', 60),
                               target_sfreq=config.get('fif_sfreq', 128.0))

    # Split into train/val/test (80/10/10) deterministically
    rng = np.random.RandomState(config.get('seed', 1234))
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    idx_train = indices[:n_train]
    idx_val = indices[n_train:n_train + n_val]
    idx_test = indices[n_train + n_val:]

    def subset(idx):
        class Subset(torch.utils.data.Dataset):
            def __init__(self, base, idxs):
                self.base = base
                self.idxs = idxs
            def __len__(self):
                return len(self.idxs)
            def __getitem__(self, i):
                x = self.base[self.idxs[i]]
                # x is tensor (C,T). Create dummy label 0 and return id
                return x, torch.tensor(0, dtype=torch.long), int(self.idxs[i])
        return Subset(dataset, idx)

    train_loader = DataLoader(dataset=subset(idx_train), batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config.get('num_workers', 0))
    val_loader = DataLoader(dataset=subset(idx_val), batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=config.get('num_workers', 0))
    test_loader = DataLoader(dataset=subset(idx_test), batch_size=config['batch_size'], shuffle=False, pin_memory=True, num_workers=config.get('num_workers', 0))

    # Fabricate Data-like shape info needed by model factory
    sample = dataset[0]
    if isinstance(sample, tuple):
        sample = sample[0]
    config['Data_shape'] = (len(idx_train), sample.shape[0], sample.shape[1])
    config['num_labels'] = 1
    return train_loader, val_loader, test_loader


def Rep_Learning(config, Data):
    # ---------------------------------------- Self Supervised Data -------------------------------------
    # If in FIF mode, build loaders from TUHFIF60sDataset
    if config.get('fif_root'):
        train_loader, val_loader, test_loader = _build_fif_dataloaders(config)
        pre_train_loader = train_loader
    else:
        if config['Pre_Training'] =='Cross-domain':
            pre_train_dataset = dataset_class(Data['pre_train_data'], Data['pre_train_label'], config['patch_size'], segment_len=config.get('segment_len', 0), split='train')
        else:
            pre_train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config['patch_size'], segment_len=config.get('segment_len', 0), split='train')
        pre_train_loader = DataLoader(dataset=pre_train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config.get('num_workers', 0))
        train_dataset = dataset_class(Data['All_train_data'], Data['All_train_label'], config['patch_size'], segment_len=config.get('segment_len', 0), split='train')
        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config.get('num_workers', 0))
        # For Linear Probing During the Pre-Training
        test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'], segment_len=config.get('segment_len', 0), split='test')
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config.get('num_workers', 0))
    # --------------------------------------------------------------------------------------------------------------
    # -------------------------------------------- Build Model -----------------------------------------------------
    logger.info("Pre-Training Self Supervised model ...")
    if not config.get('fif_root'):
        config['Data_shape'] = Data['All_train_data'].shape
        config['num_labels'] = int(max(Data['All_train_label'])) + 1
    Encoder = Encoder_factory(config)
    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    # Specify which networks you want to optimize
    networks_to_optimize = [Encoder.contex_encoder, Encoder.InputEmbedding, Encoder.Predictor]
    # Convert parameters to tensors
    params_to_optimize = [p for net in networks_to_optimize for p in net.parameters()]
    params_not_to_optimize = [p for p in Encoder.target_encoder.parameters()]

    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class([{'params': params_to_optimize, 'lr': config['lr']},
                                       {'params': params_not_to_optimize, 'lr': 0.0}])
    # scheduler: cosine with optional warmup
    warmup_epochs = int(config.get('warmup_epochs', 10))
    total_epochs = int(config['epochs'])
    # Create a simple wrapper that performs warmup by linearly increasing LR in first warmup epochs
    base_scheduler = CosineAnnealingLR(config['optimizer'], T_max=max(1, total_epochs - warmup_epochs))
    class WarmupThenCosine:
        def __init__(self, optimizer, base_scheduler, warmup_epochs, base_lr):
            self.optimizer = optimizer
            self.base_scheduler = base_scheduler
            self.warmup_epochs = warmup_epochs
            self.base_lr = base_lr
            self.last_epoch = -1
        def step(self):
            self.last_epoch += 1
            if self.last_epoch < self.warmup_epochs:
                scale = float(self.last_epoch + 1) / float(max(1, self.warmup_epochs))
                for pg in self.optimizer.param_groups:
                    if pg['lr'] > 0:
                        pg['lr'] = self.base_lr * scale
            else:
                self.base_scheduler.step()
        def get_last_lr(self):
            return [pg['lr'] for pg in self.optimizer.param_groups]
    config['scheduler'] = WarmupThenCosine(config['optimizer'], base_scheduler, warmup_epochs, config['lr'])

    config['problem_type'] = 'Self-Supervised'
    config['loss_module'] = get_loss_module()

    save_path = os.path.join(config['save_dir'], config['problem'] +'model_{}.pth'.format('last'))
    Encoder.to(config['device'])
    # ------------------------------------------------- Training The Model ---------------------------------------------
    logger.info('Self-Supervised training...')
    SS_trainer = Self_Supervised_Trainer(Encoder, pre_train_loader, train_loader, test_loader, config, l2_reg=0, print_conf_mat=False)
    SS_train_runner(config, Encoder, SS_trainer, save_path)
    # **************************************************************************************************************** #
    # --------------------------------------------- Downstream Task (classification)   ---------------------------------
    # ---------------------- Loading the model and freezing layers except FC layer -------------------------------------
    SS_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])  # Loading the model
    SS_Encoder.to(config['device'])
    # Linear probing removed for pure self-supervised training
    # print("Test ROC AUC:")
    # print(roc_auc_score(y_hat, test_labels.cpu().detach().numpy()))

    # --------------------------------- Load Data -------------------------------------------------------------
    if not config.get('fif_root'):
        train_dataset = dataset_class(Data['train_data'], Data['train_label'], config['patch_size'], segment_len=config.get('segment_len', 0), split='train')
        val_dataset = dataset_class(Data['val_data'], Data['val_label'], config['patch_size'], segment_len=config.get('segment_len', 0), split='val')
        test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'], segment_len=config.get('segment_len', 0), split='test')
        train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config.get('num_workers', 0))
        val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config.get('num_workers', 0))
        test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=config.get('num_workers', 0))

    logger.info('Starting Fine_Tuning...')
    if not config.get('fif_root'):
        S_trainer = SupervisedTrainer(SS_Encoder, None, train_loader, None, config, print_conf_mat=False)
        S_val_evaluator = SupervisedTrainer(SS_Encoder, None, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_model_{}.pth'.format('last'))
    if not config.get('fif_root'):
        Strain_runner(config, SS_Encoder, S_trainer, S_val_evaluator, save_path)

    if not config.get('fif_root'):
        best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
        best_Encoder.to(config['device'])
        best_test_evaluator = SupervisedTrainer(best_Encoder, None, test_loader, None, config, print_conf_mat=True)
        best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
        return best_aggr_metrics_test, all_metrics
    else:
        # For self-supervised only scenario, return empty metrics placeholders
        return {'loss': None}, {'total_accuracy': None}


def Supervised(config, Data):
    # -------------------------------------------- Build Model -----------------------------------------------------
    config['Data_shape'] = Data['train_data'].shape
    config['num_labels'] = int(max(Data['train_label'])) + 1
    Encoder = Encoder_factory(config)

    logger.info("Model:\n{}".format(Encoder))
    logger.info("Total number of parameters: {}".format(count_parameters(Encoder)))
    # ---------------------------------------------- Model Initialization ----------------------------------------------
    optim_class = get_optimizer("RAdam")
    config['optimizer'] = optim_class(Encoder.parameters(), lr=config['lr'], weight_decay=0)

    config['problem_type'] = 'Supervised'
    config['loss_module'] = get_loss_module()
    # tensorboard_writer = SummaryWriter('summary')
    Encoder.to(config['device'])
    # ------------------------------------------------- Training The Model ---------------------------------------------

    # --------------------------------- Load Data -------------------------------------------------------------
    train_dataset = dataset_class(Data['train_data'], Data['train_label'], config['patch_size'])
    val_dataset = dataset_class(Data['val_data'], Data['val_label'], config['patch_size'])
    test_dataset = dataset_class(Data['test_data'], Data['test_label'], config['patch_size'])

    train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    S_trainer = SupervisedTrainer(Encoder, None, train_loader, None, config, print_conf_mat=False)
    S_val_evaluator = SupervisedTrainer(Encoder, None, val_loader, None, config, print_conf_mat=False)

    save_path = os.path.join(config['save_dir'], config['problem'] + '_2_model_{}.pth'.format('last'))
    Strain_runner(config, Encoder, S_trainer, S_val_evaluator, save_path)
    best_Encoder, optimizer, start_epoch = load_model(Encoder, save_path, config['optimizer'])
    best_Encoder.to(config['device'])

    best_test_evaluator = SupervisedTrainer(best_Encoder, None, test_loader, None, config, print_conf_mat=True)
    best_aggr_metrics_test, all_metrics = best_test_evaluator.evaluate(keep_all=True)
    return best_aggr_metrics_test, all_metrics


def plot_tSNE(data, labels):
    # Create a TSNE instance with 2 components (dimensions)
    tsne = TSNE(n_components=2, random_state=42)
    # Fit and transform the data using t-SNE
    embedded_data = tsne.fit_transform(data)

    # Separate data points for each class
    class_0_data = embedded_data[labels == 0]
    class_1_data = embedded_data[labels == 1]

    # Plot with plt.plot
    plt.figure(figsize=(6, 5))  # Set background color to white
    plt.plot(class_0_data[:, 0], class_0_data[:, 1], 'bo', label='Real')
    plt.plot(class_1_data[:, 0], class_1_data[:, 1], 'ro', label='Fake')
    plt.legend(fontsize='large')
    plt.grid(False)  # Remove grid
    plt.savefig('SSL.pdf', bbox_inches='tight', format='pdf')
    # plt.show()