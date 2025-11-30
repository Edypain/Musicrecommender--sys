import os
import importlib.util

repo_root = os.getcwd()
mod_path = os.path.join(repo_root, 'api', 'recommend.py')
print('Loading module from', mod_path)
spec = importlib.util.spec_from_file_location('repo_recommend', mod_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
# call safe_load_models if present
if hasattr(mod, 'safe_load_models'):
    try:
        mod.safe_load_models()
    except Exception as e:
        print('safe_load_models raised', e)

print('DATA_LOADED=', getattr(mod, 'DATA_LOADED', None))
print('load_error_message=', getattr(mod, 'load_error_message', None))
print('similarity_matrix exists=', getattr(mod, 'similarity_matrix', None) is not None)
print('similarity shape=', None if mod.similarity_matrix is None else getattr(mod.similarity_matrix,'shape',None))
print('svd is None=', getattr(mod, 'svd', None) is None)
print('files in cwd:', os.listdir(repo_root)[:20])
print('models list:', os.listdir(os.path.join(repo_root,'models')))
