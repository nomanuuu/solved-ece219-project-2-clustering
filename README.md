Download Link: https://assignmentchef.com/product/solved-ece219-project-2-clustering
<br>
Clustering algorithms are unsupervised methods for finding groups of data points that have similar representations in a feature space. Clustering differs from classification in that no <em>a priori </em>labeling (grouping) of the data points is available.

<strong>K-means clustering </strong>is a simple and popular clustering algorithm. Given a set of data points {<strong>x</strong><sub>1</sub><em>,…,</em><strong>x</strong><em><sub>N</sub></em>} in multidimensional space, it tries to find <em>K </em>clusters such that each data point belongs to exactly one cluster, and that the sum of the squares of the distances between each data point and the center of the cluster it belongs to is minimized. If we define <em>µ<sub>k </sub></em>to be the “center” of the <em>k</em>th cluster, and

(

1<em>,     </em>if <strong>x</strong><em><sub>n </sub></em>is assigned to cluster <em>k</em>

<em>r<sub>nk </sub></em>=                                                              <em>,        n </em>= 1<em>,…,N           k </em>= 1<em>,…,K</em>

0<em>,    </em>otherwise

<em>N          K</em>

Then our goal is to find <em>r<sub>nk</sub></em>’s and <em>µ<sub>k</sub></em>’s that minimize <em>J </em>= <sup>XX</sup><em>r<sub>nk </sub></em>k<strong>x</strong><em><sub>n </sub></em>−<em>µ<sub>k</sub></em>k<sup>2</sup>. The approach

<em>n</em>=1 <em>k</em>=1

of K-means algorithm is to repeatedly perform the following two steps until convergence:

<ol>

 <li>(Re)assign each data point to the cluster whose center is nearest to the data point.</li>

 <li>(Re)calculate the position of the centers of the clusters: setting the center of the cluster to the mean of the data points that are currently within the cluster.</li>

</ol>

The center positions may be initialized randomly.

In this project, the goal includes:

<ol>

 <li>To find proper representations of the data, s.t. the clustering is efficient and gives out reasonable results.</li>

 <li>To perform K-means clustering on the dataset, and evaluate the result of the clustering.</li>

 <li>To try different preprocessing methods which may increase the performance of the clustering.</li>

</ol>

<h1>Dataset</h1>

We work with “20 Newsgroups” dataset that we already explored in <strong>Project 1</strong>. It is a collection of approximately 20,000 documents, partitioned (nearly) evenly across 20 different newsgroups, each corresponding to a different category (topic). Each topic can be viewed as a

“class”.

In order to define the clustering task, we pretend as if the class labels are not available and aim to find groupings of the documents, where documents in each group are more similar to each other than to those in other groups. We then use class labels as the ground truth to evaluate the performance of the clustering task.

To get started with a simple clustering task, we work with a well-separable portion of the data set that we used in Project 1, and see if we can retrieve the known classes. Specifically, let us define two classes comprising of the following categories.

Table 1: Two well-separated classes

<table width="707">

 <tbody>

  <tr>

   <td width="54">Class 1</td>

   <td width="653">comp.graphics comp.os.ms-windows.misc comp.sys.ibm.pc.hardware comp.sys.mac.hardware</td>

  </tr>

  <tr>

   <td width="54">Class 2</td>

   <td width="653">rec.autos                          rec.motorcycles                                 rec.sport.baseball                             rec.sport.hockey</td>

  </tr>

 </tbody>

</table>

We would like to evaluate how purely the <em>a priori </em>known classes can be reconstructed through clustering. That is, we take all the documents belonging to these two classes and perform unsupervised clustering into two clusters. Then we determine how pure each cluster is when we look at the labels of the documents belonging to each cluster.

<h1>Part 1 – Clustering of Text Data</h1>

<ol>

 <li>Building the TF-IDF matrix.</li>

</ol>

Following the steps in Project 1, <strong>transform the documents into TF-IDF vectors</strong>.

Use min df = 3, exclude the stopwords (no need to do stemming or lemmatization).

<strong>QUESTION 1: </strong>Report the dimensions of the TF-IDF matrix you get.

<ol start="2">

 <li>Apply K-means clustering with <em>k </em>= 2 using the TF-IDF data. Note that the KMeans class in sklearn has parameters named random state, max iter and n init. Please use random state=0, max iter ≥ 1000 and n init ≥ 30<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. Compare the clustering results with the known class labels. (you can refer to <a href="http://scikit-learn.org/stable/auto_examples/text/document_clustering.html">sklearn – Clustering text documents using </a><a href="http://scikit-learn.org/stable/auto_examples/text/document_clustering.html">k-means</a> for a basic work flow)

  <ul>

   <li>Given the clustering result and ground truth labels, contingency table <strong>A </strong>is the matrix whose entries <em>A<sub>ij </sub></em>is the number of data points that belong to both the class <em>C<sub>i </sub></em>the cluster <em>K<sub>j</sub></em>.</li>

  </ul></li>

</ol>

<strong>QUESTION 2: </strong>Report the contingency table of your clustering result.

<ul>

 <li>In order to evaluate clustering results, there are various measures for a given partition of the data points with respect to the ground truth. We will use the measures <strong>homogeneity score</strong>, <strong>completeness score</strong>, <strong>V-measure</strong>, <strong>adjusted Rand score </strong>and <strong>adjusted mutual info score</strong>, all of which can be calculated by the corresponding functions provided in metrics.

  <ul>

   <li><strong>Homogeneity </strong>is a measure of how “pure” the clusters are. If each cluster contains only data points from a single class, the homogeneity is satisfied.</li>

   <li>On the other hand, a clustering result satisfies <strong>completeness </strong>if all data points of a class are assigned to the same cluster. Both of these scores span between 0 and 1; where 1 stands for perfect clustering.</li>

   <li>The <strong>V-measure </strong>is then defined to be the harmonic average of homogeneity score and completeness score.</li>

   <li>The <strong>adjusted Rand Index </strong>is similar to accuracy measure, which computes similarity between the clustering labels and ground truth labels. This method counts all pairs of points that both fall either in the same cluster and the same class or in different clusters and different classes.</li>

   <li>Finally, the <strong>adjusted mutual information score </strong>measures the mutual information between the cluster label distribution and the ground truth label distributions.</li>

  </ul></li>

</ul>

<strong>QUESTION 3: </strong>Report the 5 measures above for the K-means clustering results you get.

<ol start="3">

 <li>Dimensionality reduction</li>

</ol>

As you may have observed, high dimensional sparse TF-IDF vectors do not yield a good clustering result. One of the reasons is that in a high-dimensional space, the Euclidean distance is not a good metric anymore, in the sense that the distances between data points tends to be almost the same (see [1]).

K-means clustering has other limitations. Since its objective is to minimize the sum of within-cluster <em>l</em><sub>2 </sub>distances, it implicitly assumes that the clusters are isotropically shaped, <em>i.e. </em>round-shaped. When the clusters are not round-shaped, K-means may fail to identify the clusters properly. Even when the clusters are round, K-means algorithm may also fail when the clusters have unequal variances. A direct visualization for these problems can be found at <a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py">sklearn – Demonstration of k-means assumptions</a><a href="http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_assumptions.html#sphx-glr-auto-examples-cluster-plot-kmeans-assumptions-py">.</a>

In this part we try to find a “better” representation tailored to the way that K-means clustering algorithm works, by reducing the dimension of our data before clustering.

We will use Singular Value Decomposition (SVD) and Non-negative Matrix Factorization (NMF) that you are already familiar with for dimensionality reduction.

<ul>

 <li>First we want to find the effective dimension of the data through inspection of the top singular values of the TF-IDF matrix and see how many of them are significant in reconstructing the matrix with the truncated SVD representation. A guideline is to see what ratio of the variance of the original data is retained after the dimensionality reduction.</li>

</ul>

<strong>QUESTION 4: </strong>Report the plot of the percent of variance the top <em>r </em>principle components can retain v.s. <em>r</em>, for <em>r </em>= 1 to 1000.

Hint: explained variance ratio  of TruncatedSVD objects. See <a href="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html">sklearn document</a><a href="http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html">.</a>

<ul>

 <li>Now, use the following two methods to reduce the dimension of the data. Sweep over the dimension parameters for each method, and choose one that yields better results in terms of clustering purity metrics.

  <ul>

   <li>Truncated SVD / PCA</li>

  </ul></li>

</ul>

Note that you don’t need to perform SVD multiple times: performing SVD with <em>r </em>= 1000 gives you the data projected on all the top 1000 principle components, so for smaller <em>r</em>’s, you just need to exclude the least important features.

<ul>

 <li>NMF</li>

</ul>

<strong>QUESTION 5:</strong>

Let <em>r </em>be the dimension that we want to reduce the data to (<em>i.e. </em>n components).

Try <em>r </em>= <strong>1</strong><em>,</em><strong>2</strong><em>,</em><strong>3</strong><em>,</em><strong>5</strong><em>,</em><strong>10</strong><em>,</em><strong>20</strong><em>,</em><strong>50</strong><em>,</em><strong>100</strong><em>,</em><strong>300</strong>, and plot the 5 measure scores v.s. <em>r </em>for both SVD and NMF.

Report a good choice of <em>r </em>for SVD and NMF respectively.

Note: In the choice of <em>r</em>, there is a trade-off between the information preservation, and better performance of k-means in lower dimensions.

<strong>QUESTION 6: </strong>How do you explain the non-monotonic behavior of the measures as <em>r </em>increases?

<ol start="4">

 <li>(a) Visualization.</li>

</ol>

We can visualize the clustering results by projecting the dim-reduced data points onto 2-D plane with SVD, and coloring the points according to

<ul>

 <li>Ground truth class label</li>

 <li>Clustering label respectively.</li>

</ul>

<strong>QUESTION 7: </strong>Visualize the clustering results for:

<ul>

 <li>SVD with your choice of <em>r</em></li>

 <li>NMF with your choice of <em>r</em></li>

</ul>

(b) Now try the transformation methods below to see whether they increase the clustering performance. Perform transformation on SVD-reduced data and NMF-reduced data, respectively. Still use the best <em>r </em>we had in previous parts.

<ul>

 <li>Scaling features s.t. each feature has unit variance, i.e. each column of the reduced-dimensional data matrix has unit variance (if we use the convention that rows correspond to documents).</li>

 <li>Applying a logarithmic non-linear transformation to the data vectors for the case with NMF (non-negative features):</li>

 <li>Try combining the above transformations for the NMF case.</li>

</ul>

To sum up, try the SVD case w/ and w/o performing scaling (2 possibilities). Similarly, try different combinations of w/ and w/o performing scaling and non-linarity for the NMF case (4 possibilities).

<strong>QUESTION 8: </strong>Visualize the transformed data as in part (a).

<strong>QUESTION 9: </strong>Can you justify why the “logarithm transformation” may improve the clustering results?

<strong>QUESTION 10: </strong>Report the clustering measures (except for the contingency matrix) for the transformed data.

<h1>Part 2 – Your Own Dataset</h1>

Be creative and choose an interesting dataset of your own, from your research or elsewhere. Inspired by part 1, perform clustering analysis.

<ul>

 <li>Report your pipeline and explain how you extracted meaningful features from your data, how well the clustering performed, etc.</li>

 <li>Make sure your dataset is not too trivial for a learning algorithm. Moreover, in this part, your data should include more than two classes, e.g. something between 5 and 10.</li>

</ul>

<h1>Part 3 – Color Clustering</h1>

In this part we would like to perform “segmentation” on images based on color clustering. Choose an image of size <em>m </em>× <em>n </em>of a celebrity’s face. Reshape your image to a matrix of size <em>mn </em>× 3, where the size 3 stems from the RGB channels. Transform pixel RGB information to “normalized (<em>r,g</em>) space” where:

<em>.</em>

Choose a small number of clusters, cluster the pixels according to their colors, and report your result. Here is a sample result:

Figure 1: Original figure                                Figure 2: Result of <em>k</em>-means clustering

<strong>QUESTION 11: </strong>BONUS – can you suggest a methodology to make an appropriate choice of <em>k </em>and initial seeds of cluster centers?

<a href="#_ftnref1" name="_ftn1">[1]</a> If you have enough computation power, the larger the better