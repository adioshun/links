- one epoch = one forward pass and one backward pass of all the training examples

test

- batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.

- number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

> Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.


* [딥러닝 용어 정리](http://docs.likejazz.com/deep-learning-glossary/) :  Sang-Kil Park 블로그글

test
