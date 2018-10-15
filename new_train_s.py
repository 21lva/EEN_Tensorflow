import new_models, utils
import argparse,os,imp
import tensorflow as tf
import numpy as np




parser = argparse.ArgumentParser()
parser.add_argument("-task",type=str,default="breakout")
parser.add_argument("-model",type=str,default="Stochastic")
parser.add_argument("-batch_size",type=int,default=64)
parser.add_argument("-lr",type=float,default=0.0005)
parser.add_argument("-nEpoch",type=int,default=100)
parser.add_argument("-epoch_size",type=int,default=50)
parser.add_argument("-nfeature",type=int,default=64)
parser.add_argument("-loss",type=str,default="l2",help="l1 or l2")
parser.add_argument("-gpu",type=int,default=0)
parser.add_argument("-datapath",type=str,default ="./data/",help="root directory where all the data will save")

"""
Saving Model : ./data/[task name]/[model name]/model/
Saving Log : ./data/[task name]/[model name]/log/
Saving optimizer : ./data/[task name]/[model name]/optimizer/
"""

par = parser.parse_args()

data = utils.read_data("./config.json")[par.task]

"""
Each model in json file has
    height,width,nc(#channels of images),n_actions,ncond(#input images),npred(#output images),phi_fc_size,dataloader,datapath
"""

par.samples_datapath = par.datapath+par.task+"/samples"
par.datapath = par.datapath+par.task+"/"+par.model+"/"
par.height = data["height"]
par.width = data["width"]
par.ncond = data["ncond"]
par.npred = data["npred"]
par.nc=data["nc"]
par.phi_fc_size = data["phi_fc_size"]
par.dataloader = data["dataloader"]
par.n_latent= 2 


par.filename = "task:{}-model:{}-lr:{}-epoch_size:{}-nEpoch:{}-batch_size:{}-nfeature:{}-loss:{}".format(
        par.task,par.model,par.lr,par.epoch_size,par.nEpoch,par.batch_size,par.nfeature,par.loss)
#image parsing
ImageLoader = imp.load_source("ImageLoader","dataloaders/{}.py".format(data.get("dataloader"))).ImageLoader
dataloader = ImageLoader(par)

#train , validation 
def train_epoch(model,optimizer):
    total_loss=0.0
    for _ in range(par.epoch_size):
        Input,Target ,acti= dataloader.get_batch("train")
        gradient,loss = model.compute_gradients(Input,Target)
        model.apply_gradients(gradient)
        print(loss)
        total_loss += loss
    return total_loss/par.epoch_size


def validation_epoch(model):
    total_loss =0.0
    for _ in range(par.epoch_size):
        Input,Target,act= dataloader.get_batch("valid")
        gradient,loss = model.compute_gradients(Input,Target)
        total_loss = loss
    return total_loss/par.epoch_size

def train(model,optimizer):
    os.system("mkdir -p tf"+par.datapath)
    os.system("mkdir -p tf"+par.datapath+"/model")
    os.system("mkdir -p tf"+par.datapath+"/log")

    train_losses,valid_losses =[],[]
    best_valid_loss =100000.0
    for i in range(par.nEpoch):
        train_losses.append(train_epoch(model,optimizer))
        valid_losses.append(validation_epoch(model))

        if valid_losses[-1]<best_valid_loss:
            best_valid_loss = valid_losses[-1]
            model.save_weights("tf"+par.datapath+"/model/"+par.filename+".model")
        #print log 
        logtext = "model:{} / epoch:{} / train loss:{} / validation loss:{} / best validation loss:{}".format(par.model,i,train_losses[-1],valid_losses[-1],best_valid_loss)
        utils.log(par.datapath+"/log/"+par.filename+".txt",logtext)

if __name__=="__main__":
    par.nin =par.ncond*par.nc
    par.nout = par.npred*par.nc
    
    optimizer = tf.train.AdamOptimizer(par.lr)
    model = new_models.LatentNetwork(optimizer,par)
    """
     HAVE TO WRITE DOWN ALL SENTENCES BELOW
    """

    mfile  = "task:{}-model:{}-lr:{}-epoch_size:{}-nEpoch:{}-batch_size:{}-nfeature:{}-loss:{}".format(par.task,"baseline_model",par.lr,par.epoch_size,par.nEpoch,par.batch_size,par.nfeature,par.loss)
    #baseline_model = torch.load("./data/"+par.task+"/baseline_model"+"/model/"+mfile+".model").get("model")
    model.DEncoder.load_weights("tf./data/"+par.task+"/baseline_model/model/"+mfile+".Emodel")
    model.DDecoder.load_weights("tf./data/"+par.task+"/baseline_model/model/"+mfile+".Dmodel")

    #print(baseline_model)
    #print(dir(baseline_model))

    # Model save and load
    print("==================Lets start Train=============")
    train(model,optimizer)




