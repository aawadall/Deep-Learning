Excuse my primitive knowledge. I am trying to hand-make a Deep Neural Net library, and managed somehow to define all conventional fully connected layers, with all bells and whistles related to optimizations and speedups. 

Now comes to the Computer Vision part, where I want to ingest a large image into small patches and feeding it into a convolution, filter layer.

Of course easiest way to implement this is through a for loop by:

1. re-wrap image vector into a matrix or a tensor 
1. pad the image on all 4 sides
1. loop over x and y with strides and take out patches as a sub matrix 
1. pass that matrix to a pooling layer

but that sounds a bit tedious and time consuming.

so I was thinking of vectorizing this operation to take out all patches from the vector image once using some sort of a transformation matrix, but the math is a little bit challenging. 

I was thinking of some mapping function $\mathcal{M}_k : \vec{Im} \mapsto \mathbf{P}_k$ 

where $\mathbf{P}_k \in \mathbb{R}^{s_x \cdot s_y}$ is the $k$th patch vectorized and $$ \mathbf{P} = \large [ \mathbf{P}_1 \mathbf{P}_1 \dots \mathbf{P}_k \large ] $$ an array of equally sized $k$ vectorized patches 

So $\mathcal{M}_k$ be somehow a translation matrix, copying pixels of the current patch in in the first $s_x \cdot s_y$ positions of my output, and cropping the rest of the vector

#### Example 
A $100 \times 100 \times 1$ image tensor is vectorized into a 10'000 dimensional vector, assuming no padding and a first patch that is say $3 \times 3$ pixels we want to move pixels $ \{ 1, 2, 3, 101, 102, 103, 201, 202, 203\}$ into $\mathbf{P}_1$

So I would imagine $\mathcal{M}_1 \in \mathbb{R}^{9 \times 100'000}$ be something like this 

$$ \mathcal{M}_1 = 
\begin{pmatrix}
         1 & 0 & 0 & \dots 0 & 0 & 0 & \dots 0 & 0 & 0 & \dots & 0 \\
         0 & 1 & 0 & \dots 0 & 0 & 0 & \dots 0 & 0 & 0 & \dots & 0 \\
         0 & 0 & 1 & \dots 0 & 0 & 0 & \dots 0 & 0 & 0 & \dots & 0 \\
         0 & 0 & 0 & \dots 1 & 0 & 0 & \dots 0 & 0 & 0 & \dots & 0 \\
         0 & 0 & 0 & \dots 0 & 1 & 0 & \dots 0 & 0 & 0 & \dots & 0 \\
         0 & 0 & 0 & \dots 0 & 0 & 1 & \dots 0 & 0 & 0 & \dots & 0 \\
         0 & 0 & 0 & \dots 0 & 0 & 0 & \dots 1 & 0 & 0 & \dots & 0 \\
         0 & 0 & 0 & \dots 0 & 0 & 0 & \dots 0 & 1 & 0 & \dots & 0 \\
         0 & 0 & 0 & \dots 0 & 0 & 0 & \dots 0 & 0 & 1 & \dots & 0 \\
\end{pmatrix}
$$

_note that first group of columns is at positions 1 to 3, second group is 101 to 103, and third is 201 to 203_

Ok, now we have $\mathcal{M}_1 \cdot \mathbf{Im} = \mathbf{P}_1 \in \mathbb{R}^{9 \times 1}$ 

this somehow makes sense on how to translate and crop pixels from the input image for the first patch.

### My question is:

1. How to generalize a formula to construct $\mathcal{M}_{(c_x, c_y, s_x, s_y)}$; a mapping function taking patch centered at $(c_x, c_y)$ with dimensions $s_x$ and $s_y$
2. How to stack all mapping functions $\mathcal{M}_1$ all the way to $\mathcal{M}_k$ with input image $Im$ or some tensor made of $Im$ copies to avoid looping $k$ times to construct $\mathbf{P}$

Thanks 
