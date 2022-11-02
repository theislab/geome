import pytorch_lightning as pl
import torch
import anndata as ad
from gpu_spatial_graph_pipeline.data.utils import transforms
from gpu_spatial_graph_pipeline.data.anndata2data import AnnData2DataByCategory
from gpu_spatial_graph_pipeline.data.datasets import DatasetHartmann
from gpu_spatial_graph_pipeline.models.non_linear_ncem import NonLinearNCEM
from gpu_spatial_graph_pipeline.data.datamodule import GraphAnnDataModule


fields = {
    'x': ['obs/Cluster_preprocessed', 'obs/donor'],
    'y': ['X']
}


preprocess = [
    lambda x, _: transforms.categorize_obs(x,['donor', 'Cluster_preprocessed', 'point']),
]

category_to_iterate = 'point'

a2d = AnnData2DataByCategory(
    fields=fields,
    category=category_to_iterate,
    preprocess=preprocess,
    yields_edge_index=True,
)



#Mibitof
dataset = DatasetHartmann(data_path='./example_data/hartmann/')
adatas = list(dataset.img_celldata.values())

# Merge the list of adatas and convert some string to categories as they should be
adata = ad.concat(adatas)

datas = a2d(adata)
datas


num_features = datas[0].x.shape[1]
out_channels = datas[0].y.shape[1]
num_features, out_channels

dm = GraphAnnDataModule(datas=datas, num_workers = 12, batch_size=100,learning_type='node')
model = NonLinearNCEM(
    in_channels=num_features,
    out_channels=out_channels,
    encoder_hidden_dims=[16],
    decoder_hidden_dims=[16],
    latent_dim=14,
    lr=0.0001,weight_decay=0.00001)



trainer:pl.Trainer = pl.Trainer(accelerator='gpu' if torch.torch.cuda.is_available() else 'cpu',
                                max_epochs=10,log_every_n_steps=10)

trainer.fit(model,datamodule=dm)

trainer.test(model, datamodule=dm)