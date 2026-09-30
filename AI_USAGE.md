# ITCS-6169-CNNChallenge-AndyReed
Your goal is to build a high-performing CNN-based image classifier from a small training dataset of only 2,400 images across 16 classes.
# AI Usage

## AI Tools Used

I used ChatGPT as a coding and debugging assistant during the development
of this assignment. AI assistance was used to explain PyTorch errors,
suggest code modifications, review experimental results, and help organize
the experimental process. I reviewed and tested AI-generated suggestions
before incorporating them into the project.

## Representative Examples

### 1. Converting the Baseline from Grayscale to RGB

The starter network and preprocessing pipeline originally operated on
single-channel grayscale images. I used ChatGPT to help modify the data
pipeline to preserve RGB information and introduce training-only data
augmentation.

After this change, the input tensors changed from:

[N, 1, 64, 64]

to:

[N, 3, 64, 64]

This produced a PyTorch channel mismatch because the first convolution
still expected one input channel. AI assistance helped identify the source
of the error and suggested changing the first convolution from one input
channel to three.

I verified the change by checking that the input batch had shape
[64, 3, 64, 64] and that the network produced logits with shape
[64, 16].

### 2. Separating Training and Validation Transformations

AI assistance was used to identify a problem with applying random
augmentation through a single ImageFolder object shared by both training
and validation subsets.

The data pipeline was modified so that training and validation used
separate ImageFolder instances with identical image indices but different
transformations. Random horizontal flipping and rotation were applied only
to training data, while validation preprocessing remained deterministic.

I verified that both subsets used the same reproducible 80/20 split and
that only the training transformation contained random augmentation.

### 3. Designing and Debugging the Deeper CNN

I used ChatGPT to discuss a deeper CNN architecture after the starter TNet
showed limited validation performance. The resulting architecture used
multiple convolutional layers, progressively increasing channels from
32 to 64 to 128, followed by adaptive average pooling and dropout.

I verified the architecture by checking the tensor dimensions before
training. A batch with shape [64, 3, 64, 64] produced output logits with
shape [64, 16], matching the 16 scene categories.

### 4. Learning-Rate Scheduling

After the deeper CNN reached only 51.46% validation accuracy after
20 epochs, AI assistance was used to implement ReduceLROnPlateau and
modify the training function to record the learning rate.

The first AI-assisted implementation contained an error involving the
scheduler argument and duplicate versions of the training function.
This caused Python to report that the training function received multiple
values for the epochs argument.

I inspected the function signature, removed the duplicate definition,
and changed the function to accept generic optimizer and scheduler
arguments. I verified the correction by confirming that training ran for
40 epochs and printed the learning rate during every epoch.

### 5. Interpreting Experiment Results

AI assistance was used to discuss the loss and validation-accuracy curves,
but I used the observed experimental results to determine which experiment
to run next.

Experiment 2 initially appeared unsuccessful because its best validation
accuracy after 20 epochs was 51.46%, below the 53.96% obtained in
Experiment 1. However, I observed that both training and validation losses
were still generally decreasing. I therefore decided to retain the deeper
architecture and train it longer rather than immediately replacing it.

With 40 epochs and learning-rate scheduling, the same deeper CNN reached
64.79% validation accuracy.

## Incorrect or Questionable AI Suggestion

One AI-assisted version of the scheduler training function used
inconsistent parameter names and was later duplicated in the notebook.
This resulted in the error:

TypeError: train_model_with_scheduler() got multiple values for argument
'epochs'

Rather than changing the experiment configuration to work around the
error, I inspected the active Python function signature. This showed that
the scheduler parameter was missing from the active function definition.

I corrected the function so that it explicitly accepted:

model, train_loader, val_loader, optimizer, scheduler, epochs, device

and verified the function signature before restarting training.

## Human Experimental Decision

An important decision I made was not to abandon the deeper CNN after
Experiment 2.

Its validation accuracy of 51.46% was lower than Experiment 1's 53.96%,
so selecting models only from the final accuracy numbers could have led
me back to the simpler network. Instead, I examined the training and
validation loss curves and concluded that the deeper network appeared
undertrained rather than severely overfit.

I therefore kept the architecture and tested longer training with
learning-rate scheduling. Experiment 3 reached a best validation accuracy
of 64.79%, substantially outperforming the earlier experiments.

This decision was based on my interpretation of the experimental evidence
rather than simply accepting an AI recommendation.
