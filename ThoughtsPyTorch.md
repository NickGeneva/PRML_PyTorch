# Toughts on PyTorch

This document is essentially a brief reflection of what I saw while implmenting the algorithms discussed in Bishop's book in Python and slowing using PyTorch where I felt necessary.

### Chapters 1-3
While I am well aware that pyTorch has many trend fitting functions already built into it I did not take advantage of any these simply because I wanted to make sure I understood the math behind each algorithm. Instead the for was on pyTorch's **Tensor** datatype.

Basically Tensors are a matrix datatype that I would argue is more restrictive than other array storage methods, however Tensors are a MUST for GPU calculations. So it is essential that you use them. It was important to note that Tensor's default datatype is a *single percision* float. For most machine learning applications *double precision* is a must so be sure to always upgrade your tensor to doubles with **.type(th.DoubleTensor)**.

### Chapter 4
Chapter 4 was the point at which I started to play around with the autograd features in PyTorch using it to auto differentiate my loss functions in the classification. I should note that I also implmeneted the explicit version as well. The major take away from this is PyTorch is **SLOOOOOWWWW** compared to just doing the derivative and manually writing it out in the program. It was at least 3 times slower if not more and when problems get big that is going to be an issue.

Sure I understand that sometime getting the derivative is not always easy or even possible but speed should always be considered.

### Chapter 5
Chapter 5 is the dive into PyTorch's neural net library or *torch.nn*. I gotta say it is actually pretty easy to set up once you know what the heck you're doing. The best way to learn is by example, period. I think fig5-9.py is a really nice example of how to set up a simple NN using pyTorch's modules.

However I do have concerns for the future. While PyTorch was great now with a simple 2-layer NN what happens when stuff gets more complicated? What if I need to get the Hessian? I just have worries about the difficulties of getting under the hood of PyTorch easily. However this could easily be attributed to me not being super well versed with the library yet.
