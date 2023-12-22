import keras_deep_learning
import keras_deep_learning.projects.pokemon_classifier as pc
import os, shutil


def get_categories():
    categories = next(os.walk(orig_data_dir))[1]
    categories.sort()
    return categories


def get_num_train_samples():
    return _get_num_samples(train_dir)


def get_num_val_samples():
    return _get_num_samples(val_dir)


def _get_num_samples(dir_path):
    num_samples = 0
    for category in get_categories():
        c_path = os.path.join(dir_path, category)
        num_samples += len(next(os.walk(c_path))[2])
    return num_samples


def load_data():
    if os.path.exists(train_dir):
        print('Train directory already exists. Deleting...')
        shutil.rmtree(train_dir)
    os.mkdir(train_dir)
    if os.path.exists(val_dir):
        print('Validation directory already exists. Deleting...')
        shutil.rmtree(val_dir)
    os.mkdir(val_dir)
    if os.path.exists(test_dir):
        print('Test directory already exists. Deleting...')
        shutil.rmtree(test_dir)
    os.mkdir(test_dir)

    categories = get_categories()

    category_dst_dirs = {c : tuple([os.path.join(d, c) for d in [
        train_dir, val_dir]]) for c in categories}

    for category, dst_dirs in category_dst_dirs.items():
        src_dir = os.path.join(orig_data_dir, category)
        num_samples = len(next(os.walk(src_dir))[2])
        num_train_samples = round(num_samples * 0.8)
        sample_ranges = (range(num_train_samples),
            range(num_train_samples, num_samples))
        for dst_dir, s_range in zip(dst_dirs, sample_ranges):
            os.mkdir(dst_dir)
            fnames = next(os.walk(src_dir))[2]
            fnames.sort(key=lambda f: int(f.split('.')[1]))
            for fname in fnames:
                _, index, _ = tuple(fname.split('.'))
                if int(index) in s_range:
                    src = os.path.join(src_dir, fname)
                    dst = os.path.join(dst_dir, fname)
                    print('Copying', fname, 'to', dst)
                    shutil.copyfile(src, dst)


orig_data_dir = keras_deep_learning.__file__.replace(
    '__init__.py', 'datasets\\pokemon_gen1')
dst_base_dir = pc.__file__.replace(
    '__init__.py', 'data')
train_dir = os.path.join(dst_base_dir, 'train')
val_dir = os.path.join(dst_base_dir, 'validation')
test_dir = os.path.join(dst_base_dir, 'test')

if __name__ == '__main__':
    load_data()