import yaml
import io
import numpy as np

def read_config(config_path):
    with open(config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content

def logging_model_summary(model):
    with io.StringIO() as stream:
        model.summary(print_fn= lambda x: stream.write(f"{x}\n"))
        summary_lines = stream.getvalue()
    return summary_lines

#Updating labels 
def update_even_odd_labels(list_of_labels):
    for idx,label in enumerate(list_of_labels):
        even_check = label%2 == 0
        list_of_labels[idx] = np.where(even_check ,1,0)
    return list_of_labels