from models.ira_unet import base_Unet

def get_model(model_name:str, model_args:dict):
    if model_name == 'IRA_Unet':
        return base_Unet(**model_args)
    
    else: 
        raise ValueError(f'Model name {model_name} is not valid.')

if __name__ == '__main__':
    pass
