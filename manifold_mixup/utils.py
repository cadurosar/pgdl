
import tensorflow as tf
try:
    import tqdm
    def progress_bar(num_batchs):
        return tqdm.tqdm(range(num_batchs), leave=False, ascii=True)
except:
    def progress_bar(num_batchs):
        return range(num_batchs)


@tf.function
def partial_forward(model, x, layer_cut):
    for layer in model.layers[:layer_cut]:
        x = layer(x)
    return x

def resume_model(model, layer_cut):
    @tf.function
    def f(x):
        for layer in model.layers[layer_cut:]:
            x = layer(x)
        return x
    return f
