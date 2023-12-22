import keras_deep_learning
import os


dataset_dir_path = keras_deep_learning.__file__.replace(
    '__init__.py', 'datasets\\pokemon_gen1\\')

extension_counts = {}

total_failed = 0
pkm_dir_names = next(os.walk(dataset_dir_path))[1]
for pkm_name in pkm_dir_names:
    pkm_dir_path = os.path.join(dataset_dir_path, pkm_name)
    fnames = next(os.walk(pkm_dir_path))[2]
    failed = 0
    for i, fname in enumerate(fnames):
        f_ext = list(fname.split('.'))[-1]
        if f_ext not in extension_counts:
            extension_counts[f_ext] = 0
        extension_counts[f_ext] += 1
        src_name = os.path.join(pkm_dir_path, fname)
        if f_ext not in []:
            new_fname = ('%s.%s.%s' % (pkm_name, i - failed, f_ext))
            dst_name = os.path.join(pkm_dir_path, new_fname)
            print('Renaming', fname, 'to', new_fname)
            os.rename(src_name, dst_name)
        else:
            print('Removing', src_name)
            os.remove(src_name)
            failed += 1
            total_failed += 1
print(extension_counts)
print(total_failed, 'failed')