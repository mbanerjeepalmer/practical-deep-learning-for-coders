# Chapter 3

Very interesting actually: https://github.com/fastai/fastbook/blob/master/03_ethics.ipynb
And disappointingly telling that even Jeremy Howard doesn't include it on his main video series. I only noticed it because of the book.
## Questions
1. Does ethics provide a list of "right answers"?
2. How can working with people of different backgrounds help when considering ethical questions?
3. What was the role of IBM in Nazi Germany? Why did the company participate as it did? Why did the workers participate?
4. What was the role of the first person jailed in the Volkswagen diesel scandal?
5. What was the problem with a database of suspected gang members maintained by California law enforcement officials?
6. Why did YouTube's recommendation algorithm recommend videos of partially clothed children to pedophiles, even though no employee at Google had programmed this feature?
7. What are the problems with the centrality of metrics?
8. Why did Meetup.com not include gender in its recommendation system for tech meetups?
9. What are the six types of bias in machine learning, according to Suresh and Guttag?
10. Give two examples of historical race bias in the US.
11. Where are most images in ImageNet from?
12. In the paper ["Does Machine Learning Automate Moral Hazard and Error"](https://scholar.harvard.edu/files/sendhil/files/aer.p20171084.pdf) why is sinusitis found to be predictive of a stroke?
13. What is representation bias?
14. How are machines and people different, in terms of their use for making decisions?
15. Is disinformation the same as "fake news"?
16. Why is disinformation through auto-generated text a particularly significant issue?
17. What are the five ethical lenses described by the Markkula Center?
18. Where is policy an appropriate tool for addressing data ethics issues?

# Chapter 4

## Key concepts

- We first set create a baseline model.
	- In creating the model we combine our many tensors (which may be three-dimensional already? actually no, they are two-dimensional with a value) into a single, three-dimensional (rank-3) tensor. We use PyTorch's `stack` to do this.
	- disambiguation of rank, axis, and length
	- L1 norm (mean absolute difference) vs L2 norm (RMSE)
- PyTorch tensors vs Numpy arrays
	- A 'jagged array' is one where the arrays within an array are of different sizes.
	- PyTorch tensors don't allow jagged arrays. It only supports regularly-shaped multidimensional rectangular structures. PyTorch allows data structures to live on the GPU.
- Broadcasting
	- Increases the rank of the lower-rank tensor.
	- By this point we:
		- used a simple pair of ideals for 3 and 7
		- compare candidate against the 3 and 7 and pick which is the most similar
		- use broadcasting to efficiently compare all of these items from the validation set
		- define accuracy as the proportion of the time that's correct with the validation set
- Stochastic Gradient Descent
	- Weights...
	- Steps:
		- Initialise weights
		- Predict for each image
		- Loss (how good were its predictions)
		- Gradient (how would changing certain weights influence loss)
		- Step (i.e. change the weights, using the gradient)
		- Repeat prediction
		- Iterate until you stop
	- Definitions:
		- `x` is the value you have
		- `y` is the prediction
- MNIST
	- We need to define a new loss function (confusing earlier that the loss function seems to be the same throughout?). But either way now we're defining a new one.
	- 'We go from a list of matrices (rank-3 tensor) to a list of vectors (rank-2 tensor).'
		- So we have `stacked_threes` and `stacked_sevens` which have shape `torch.Size([6131, 28, 28])`.
		- So now it's all the images with all the pixels in one long vector: `torch.Size([12396, 784])`.
	- > 'The function `weights*pixels` won't be flexible enough'
		- This confused me.
		- But I suppose when we think about it, we're saying that if the pixel in some bottom middle position has a very large value and we multiply it by the weight, which let's say is 0.8 then we end up with a number close to 1, which means it is likely to be a 3 rather than 7. Whereas if the `x` value is actually very low then it's more likely to be a 7 so the `y` value will be...low as well? I suppose with a very big `w`eight then we always get a big number. (I think it says later that we deal with this using a special equation.)
	- Then it introduces the bias
		- (I didn't get an intuitive understanding of why this was necessary other than 'otherwise zeroes will always be zeroes'.)
	- So then we go and run a prediction
		- `(train_x[0]*weights.T).sum() + bias`
		- `tensor([-8.7777], grad_fn=<AddBackward0>)`
		- This is below zero so apparently a prediction of a 7.
		- And then `corrects = (preds>0.0).float() == train_y` tells us whether this list of 1's and 0's is the same as the labels of 1's and 0's that are in our training set.
	- Calculate gradients
		- I actually had the impression we could get gradients before making step changes.
		- And now it makes sense why he shows us that the accuracy is exactly the same even after making a small change the loss doesn't change.
	- Calculating new loss function
		- We can hand-write a new loss function that compares the distance between each prediction and the actual targets.
		- Sigmoid
			- Ensures all values in a range are smooshed onto an output value between 0 and 1. And the range is on a smooth increasing curve.
		- Loss vs accuracy:
			- > The key difference is that the metric is to drive human understanding and the loss is to drive automated learning. To drive automated learning, the loss must be a function that has a meaningful derivative. It can't have big flat sections and large jumps, but instead must be reasonably smooth. This is why we designed a loss function that would respond to small changes in confidence level. This requirement means that sometimes it does not really reflect exactly what we are trying to achieve, but is rather a compromise between our real goal and a function that can be optimized using its gradient. The loss function is calculated for each item in our dataset, and then at the end of an epoch the loss values are all averaged and the overall mean is reported for the epoch.
			- > Metrics, on the other hand, are the numbers that we really care about. These are the values that are printed at the end of each epoch that tell us how our model is really doing. It is important that we learn to focus on these metrics, rather than the loss, when judging the performance of a model.
		- SGD and mini-batches
			- We need to update weights based on gradients. That is, an image has passed through the model. The model has made a prediction about which number has been handwritten. It's wrong for that item, to some degree for each weight (the loss).
			- Calculating for the whole dataset at once takes too long (but maybe would be a good reflection of the average loss?).
			- So we take a number of items in a mini batch.
			- `DataLoader` and `Dataset`
				- `DataLoader` Shuffles and minibatches the data.
				- `Dataset` contains tuples of independent and dependent variables.
				- So passing a `Dataset` to a `DataLoader` we get mini-batches of tuples of tensors, where each tensor is a batch of independent or dependent variables.
	- Putting it all together
		- replacing linear1 with nn.linear
		- epochs?
			- Well there's `train_epoch`. And that says that for each pair of value and label in the `DataLoader` (which was globally defined before)
		- Creating an optimiser
			- We see `BasicOptim` is just the two functions for `step` and `zero_grad`.
- Adding a Nonlinearity
	- 

I absolutely hate Howard's continued use of global variables.

```python
def train_epoch(model, lr, params):
    for xb,yb in dl:
        calc_grad(xb, yb, model)
        for p in params:
            p.data -= p.grad*lr
            p.grad.zero_()
```

```python
Sequential(
  (0): Linear(in_features=784, out_features=30, bias=True)
  (1): ReLU()
  (2): Linear(in_features=30, out_features=1, bias=True)
)
```

## To revise
- [ ] What is a tensor?
	- "We already know how to create a tensor containing a single image." Assume this was from an earlier chapter.
	- Maybe a tensor is just a matrix.
- [ ] How do these various functions in the SGD step work?
	- When we are trying to reduce the loss for each tensor, if `x` is at `3.0` but we want it at `0.0`, why is it `0.0`? I suppose that's the loss function, so we always want the loss closer to `0.0`.
	- But that means:
		- We have a lot of different weights. When feeding these weights as parameters to a function we end up with some values (predictions).
		- Then the evaluation step compares the predictions with the actuals. The difference is called the 'loss'.
		- We can use derivatives to understand how the weight value ought to be adjusted to get the loss closer to zero. That's the slope of the derivative. (Actually, is the gradient really two-dimensional? Surely given it's just a number it's saying 'make it bigger' or 'make it smaller'. And actually the fact the number is bigger or smaller is the hint that it is further or closer.)
- [ ] Derivatives (Khan Academy/3B1B)
- [ ] Matrix multiplication (Khan Academy/3B1B)
- [ ] How exactly does `train_epoch` work, line by line?
	- For example
