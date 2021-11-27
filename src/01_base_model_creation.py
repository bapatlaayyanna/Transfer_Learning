from src.utils import dataset, common, logging
import argparse
import os
import tensorflow as tf

log_file = open('logs/base_model_training.txt','a+')

def training(config_path):
    config = common.read_config(config_path)

    #Getting mnist datset
    logging.log(log_file,'Getting mnist data from get_dataset method')
    x_train,y_train,x_valid,y_valid,x_test,y_test = dataset.get_dataset()
    logging.log(log_file, 'Successfully got the mnist dataset from get_dataset method')

    #setting seed values
    logging.log(log_file,'Setting seed values')
    seed_value = config['params'] ['SEED']
    dataset.setting_seed(seed_value=seed_value)
    logging.log(log_file,'seed value set successfully')

    #define LAYERS
    logging.log(log_file,'Intialising layers for model creation')
    LAYERS = dataset.get_layers()
    logging.log(log_file,'Layers Initialization is over')

    #Creating model and compiling
    model = tf.keras.models.Sequential(LAYERS)
    LOSS = config['params']['LOSS']
    OPTIMIZER = config['params']['OPTIMIZER']
    METRICS = config['params']['METRICS']
    model.compile(loss =LOSS, optimizer= OPTIMIZER, metrics = [METRICS])

    #writing model summary in log file
    model_summary = common.logging_model_summary(model)
    logging.log(log_file,f'model summary is ######## \n {model_summary}')

    #training the model
    logging.log(log_file,'Model training is started')
    history = model.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs = config['params']['epochs'], verbose = 2)
    logging.log(log_file,'Model training is over')

    #saving model in artifacts directory
    logging.log(log_file,'Saving Model in artifcats directory')
    model_dir = os.path.join('artifacts','models')
    os.makedirs(model_dir,exist_ok=True)
    model_name = os.path.join(model_dir,'base_model.h5')
    model.save(model_name)
    logging.log(log_file,f'Model is saved at {model_name}')

    logging.log(log_file,f'Model accuracy metrics is :{model.evaluate(x_test,y_test)}')



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config','-c',default = 'configs/config.yaml')
    pased_args = args.parse_args()

    try:
        logging.log(log_file, 'Training started')
        training(config_path=pased_args.config)
        logging.log(log_file, 'Training Ended')

    except Exception as e:
        logging.log(log_file,'Error occured during training:'+str(e))
        raise e
    
    

