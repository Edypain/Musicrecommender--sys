import os
import api.recommend as recommend

try:
    recommend.safe_load_models()
except Exception as e:
    print('safe_load_models raised:', e)

print('CWD=', os.getcwd())
print('models dir exists=', os.path.exists(os.path.join(os.getcwd(),'models')))
print('similarity exists=', os.path.exists(os.path.join(os.getcwd(),'models','similarity_matrix.npy')))
print('svd exists=', os.path.exists(os.path.join(os.getcwd(),'models','svd.pkl')))
print('dataset exists=', os.path.exists(os.path.join(os.getcwd(),'music_dataset.csv')))
print('DATA_LOADED=', recommend.DATA_LOADED)
print('load_error_message=', recommend.load_error_message)
print('similarity shape=', None if recommend.similarity_matrix is None else getattr(recommend.similarity_matrix,'shape',None))
print('svd is None=', recommend.svd is None)
