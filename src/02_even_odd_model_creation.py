from src.utils import dataset, common, logging
import argparse
import os
import tensorflow as tf

log_file = open('logs/Even_odd_model_training.txt','a+')

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
    logging.log(log_file,'Loading base model')
    model_path = os.path.join('artifacts', 'models', 'base_model.h5')
    base_model = tf.keras.models.load_model(model_path)

    logging.log(log_file,'Setting output layers weights as trinable weights')
    for layer in base_model.layers[:-1]:
        layer.trainable = True
    base_model_layers = base_model.layers[:-1]
    logging.log(log_file,'Assiging base model layers to our new model')
    even_model = tf.keras.models.Sequential(base_model_layers)

    even_model.add(tf.keras.layers.Dense(2,activation = "softmax", name = 'outputlayer'))


    #Creating model and compiling
    
    LOSS = config['params']['LOSS']
    OPTIMIZER = config['params']['OPTIMIZER']
    METRICS = config['params']['METRICS']
    even_model.compile(loss =LOSS, optimizer= OPTIMIZER, metrics = [METRICS])

    #writing model summary in log file
    even_model_summary = common.logging_model_summary(even_model)
    logging.log(log_file,f'Even/odd model summary is ######## \n {even_model_summary}')

    #training the model
    logging.log(log_file,'Model training is started')
    y_train_even, y_valid_even, y_test_even = common.update_even_odd_labels([y_train,y_valid,y_test])
    history = even_model.fit(x_train, y_train_even, validation_data = (x_valid, y_valid_even), epochs = config['params']['epochs'], verbose = 2)
    logging.log(log_file,'Model training is over')

    #saving model in artifacts directory
    logging.log(log_file,'Saving Model in artifcats directory')
    model_dir = os.path.join('artifacts','models')
    os.makedirs(model_dir,exist_ok=True)
    model_name = os.path.join(model_dir,'even_odd_model.h5')
    even_model.save(model_name)
    logging.log(log_file,f'Model is saved at {model_name}')

    logging.log(log_file,f'Even/Odd Model accuracy metrics is :{even_model.evaluate(x_test,y_test_even)}')



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config','-c',default = 'configs/config.yaml')
    pased_args = args.parse_args()

    try:
        logging.log(log_file, 'Even model Training started')
        training(config_path=pased_args.config)
        logging.log(log_file, 'Even model Training Ended')

    except Exception as e:
        logging.log(log_file,'Error occured during training:'+str(e))
        raise e
    
    

