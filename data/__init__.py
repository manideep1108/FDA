import numpy as np
from torch.utils import data
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
from data.cityscapes_dataset_label import cityscapesDataSetLabel
from data.cityscapes_dataset_SSL import cityscapesDataSetSSL
from data.synthia_dataset import SYNDataSet
from data.Cityscapes_SRC_Dataset import Cityscapes_SRC_Dataset
from data.ACDC_TRG_Dataset import ACDC_TRG_Dataset

IMG_MEAN = np.array((0.0, 0.0, 0.0), dtype=np.float32)
image_sizes = {'cityscapes': (1024,512), 'gta5': (1280, 720), 'synthia': (1280, 760)}
cs_size_test = {'cityscapes': (1344,576)}

def CreateSrcDataLoader(args):

    source_dataset = Cityscapes_SRC_Dataset(args.img_dir_src,
                                            args.label_dir_src,
                                            crop_size=(1024, 512),
                                            mean=IMG_MEAN,
                                            set= 'train')
    source_dataloader = data.DataLoader( source_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True )
    try:
        while True:
            return source_dataloader
    except StopIteration:
            pass
            
        
    

def CreateTrgDataLoader(args):

    target_dataset = ACDC_TRG_Dataset(args.img_dir_trg,
                                            args.label_dir_trg,
                                            crop_size=(1024, 512),
                                            mean=IMG_MEAN,
                                            set='train')
    target_dataloader = data.DataLoader(target_dataset,
                                        batch_size=args.batch_size,
                                        shuffle=True,
                                        num_workers=args.num_workers,
                                        pin_memory=True)
    try:
        while True:
            return target_dataloader
    except StopIteration:
            pass




def CreateTrgDataSSLLoader(args):
    target_dataset = cityscapesDataSet( args.data_dir_target, 
                                        args.data_list_target,
                                        crop_size=image_sizes['cityscapes'],
                                        mean=IMG_MEAN, 
                                        set=args.set )
    target_dataloader = data.DataLoader( target_dataset, 
                                         batch_size=1, 
                                         shuffle=False, 
                                         pin_memory=True )
    try:
        while True:
            return target_dataloader
    except StopIteration:
            pass



def CreatePseudoTrgLoader(args):
    target_dataset = cityscapesDataSetSSL( args.data_dir_target,
                                           args.data_list_target,
                                           crop_size=image_sizes['cityscapes'],
                                           mean=IMG_MEAN,
                                           max_iters=args.num_steps * args.batch_size,
                                           set=args.set,
                                           label_folder=args.label_folder )

    target_dataloader = data.DataLoader( target_dataset,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.num_workers,
                                         pin_memory=True )

    try:
        while True:
            return target_dataloader
    except StopIteration:
            pass

