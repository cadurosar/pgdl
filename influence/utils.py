
import tensorflow as tf
try:
    import tqdm
    def progress_bar(num_batchs):
        return tqdm.tqdm(range(num_batchs), leave=False, ascii=True)
except:
    def progress_bar(num_batchs):
        return range(num_batchs)

def resume_model(model, layer_cut):
    @tf.function
    def f(x):
        for layer in model.layers[layer_cut:]:
            x = layer(x)
        return x
    return f


##########################################
############ DATASETS LOADING ############
##########################################


def balanced_batchs(dataset, num_labels, batch_size):
    classes = []
    for label in range(num_labels):
        cur_class = dataset.filter(lambda data, y: tf.math.equal(y, label))
        cur_class = cur_class.repeat().shuffle(256)
        classes.append(cur_class)
    return tf.data.experimental.sample_from_datasets(classes).batch(batch_size)

def raw_batchs(dataset, batch_size):
    return dataset.repeat().shuffle(buffer_size=10000).batch(batch_size)