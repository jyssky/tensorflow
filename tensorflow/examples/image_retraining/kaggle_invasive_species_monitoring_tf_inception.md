# Retrained TensorFlow Inception Model for Kaggle competition 'Invasive Species Monitoring'

This document describes my implementations for Kaggle competition [Invasive Species Monitoring](https://www.kaggle.com/c/invasive-species-monitoring), followed by discussions of techniques for performance improvement.

## Implementations
I retrained last layers of Inecption module based on TensowFlow [retrain.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py). With fine tuned hyperparameters and some modifications of the last layers, the working version includes:

* [retrain.py](https://github.com/jyssky/tensorflow/blob/invasive_species_monitoring/tensorflow/examples/image_retraining/retrain.py): the script to retrain the last layers of the model
* [label_image.py](https://github.com/jyssky/tensorflow/blob/invasive_species_monitoring/tensorflow/examples/image_retraining/label_image.py): the script to generate a csv file containing results for submission

To retrain the model, run the following command:
```
python tensorflow/examples/image_retraining/retrain.py \
  --image_dir=/home/jys/Documents/kaggle/invasive_species_monitoring/train \
  --learning_rate=0.001 \
  --summaries_dir=<directory_to_summary_dir> \
  --how_many_training_steps=500 \
  --keep_prob=0.7 \
  --pos_weight=0.4 \
  --output_graph=<path_of_output_graph> \
  --train_batch_size=50 \
  --print_misclassified_test_images \
  --train_lables_path=<path_to_train_label>
```

To generate results for submission, run the following command
```
python tensorflow/examples/image_retraining/label_image.py \
  --graph=<path_of_output_graph> \
  --csv=<path_of_csv_containing_results_for_submission> \
  --test_image_dir=<directory_of_test_images>
```

## Discussion of techniques

The above scripts generated results with 0.976 accuracy. Below is a short summary of techniques that I have tried:
- Dropout layer: Dropout is a good choise of Regularizer for Neural networks to prevent overfitting, especially since there are only about 2000 training images.
- Relu after last dense layer: I added ReLu layer after the dense layers, but it did not help.
- Adam optimizer: With the GradientDecent Optimizer and the default training hyperparameters in TensowFlow [retrain.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py), the trained models can achieve around 0.90 accuracy. With Adam optimizer (with fine tuned parameters), the model can achieve 0.97 accuracy.
- Weighting of positive errors: The most common loss function for Neural Networks is softmax_cross_entropy_with_logits. However for this problem, it is a binary classification method, e.g., whether the image contains `invasive species`. Then I changed to use [weighted_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits), which allows an extra parameter (`pos_weight`) to trade off recall and precision. It can be proven that `sigmoid_cross_entropy_with_logits` is equivalent to [softmax_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits) for binary classification problems.

As learned from the [1st place solution](https://www.kaggle.com/c/invasive-species-monitoring/discussion/38165), here are the techniques I should try:
- Training data improvement: I should have used wider range of image sizes (larger than 299x299 dimension) and crops of different parts of the images for training data.
- Ensembling: I should have averaged the predictions from various trained models. Even with the same hyperparameters, the trained models will end up with different weightings. So, different trained models grasp different features of the images, and ensembling improves performance.