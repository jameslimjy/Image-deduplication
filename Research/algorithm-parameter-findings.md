# algorithm-parameter-findings
The purpose of this markdown file is to document the findings from changing parameters of the image-deduplicate-algorithm with the iPhone gallery images dataset. To recap, there are 2 parameters that the image-deduplicate-algorithm has - **explained_var** and **hyperspace_distance_percentage**. 

## Table Of Contents
1. [Parameters](#parameters)
    1. [explained_var](#explained_var)
    2. [hyperspace_distance_percentage](#hyperspace_distance_percentage)
2. [iPhone Gallery Dataset](#iphone-gallery-dataset)
3. [Macro Metrics](#macro-metrics)
    1. [FMI](#fmi)
    2. [Number Of Clusters](#number-of-clusters)
    3. [Conclusion](#macro-metrics-conclusion)
4. [Visual Inspection Of Different Parameter Combinations](#visual-inspection-of-different-parameter-combinations)
    1. [explained_var 0.98, hyperspace_distance_percentage 10%](#ev0.98-hdp0.1)
    2. [explained_var 0.98, hyperspace_distance_percentage 5%](#ev0.98-hdp0.05)
    3. [explained_var 0.98, hyperspace_distance_percentage 15%](#ev0.98-hdp0.15)
    4. [explained_var 0.9, hyperspace_distance_percentage 10%](#ev0.9-hdp0.1)
    5. [explained_var 0.8, hyperspace_distance_percentage 10%](#ev0.8-hdp0.1)
    6. [Conclusion](#diff-parameter-combination-conclusion)

## Parameters <a name="parameters"></a>
### explained_var <a name="explained_var"></a>
PCA is used to reduce the dimensionality of a dataset. As discussed in [image-deduplicate-algorithm-walkthrough.py](https://github.com/tictag-io/image-deduplicate/blob/main/Research/image-deduplicate-algorithm-walkthrough.py), we can select the number of components by specifying a float that represents the amount of variance that will be explained by the number of components chosen. For example, if explained_var = 0.9 results in 300 principal components being chosen, explained_var = 0.95 will definitely result in >300 principal components since the PCA object will need to select more principal components to account for greater explained variation.

### hyperspace_distance_percentage <a name="hyperspace_distance_percentage"></a>
This parameter is used to determine what epsilon (eps) value will be used in the DBSCAN clustering algorithm. eps is defined as the maximum distance between two samples for one to be considered as in the neighborhood of the other. Since this value is arbitarily determined and will vary largely based on dataset used, a better approach would be to derive the value of eps from the hyperspace diameter value. 

After the feature vectors have been transformed via PCA to new condensed feature vectors, we find the diameter of this new hyperspace and use the user-input hyperspace_distance_percentage to calculate what eps should be. For example, if the diameter of the hyperspace is 100 and the user input a hyperspace_distance_percentage value of 5%, the eps value that is used for the DBSCAN clustering algorithm is 5.

## iPhone Gallery Dataset <a name="iphone-gallery-dataset"></a>
This dataset consists of **600 images** taken from an iPhone gallery, this dataset is not publicly available as it contains personal images. This dataset contains several near duplicate images (from burst shots) and has a wide variety of image content ranging from selfies to screenshots to images of scenery. This dataset has also been labelled - manual visual inspection of the photos was conducted and photos deemed to be similar looking are recorded as clusters. Of all the images, there were **191 images** that were deemed to belong to a cluster and a total of **73 clusters** identified.


## Macro Metrics <a name="macro-metrics"></a>
As discussed in image-deduplicate-evaluation.py, there are different metrics that can be used to evaluate how good a clustering is. Since the iPhone gallery image dataset is labelled, we can make use of the Fowlkes Mallow Index score (FMI). Before diving deeper into the findings of using different combinations of parameters, lets first explore the relationship between changing the parameters (explained_var and hyperspace_distance_percentage) and the FMI score. While we're at it, we'll also look at the number of clusters formed.

<p align="center">
    <img width="750" alt="Relationship between parameters and macro metrics" src="https://user-images.githubusercontent.com/56946413/129138086-0a5552af-781d-43a4-bd9b-973e2387b368.png">
</p>
<p align="center">Relationship between parameters and macro metrics</p>

The figure above shows the relationship between the various combinations of parameters plot against FMI and Number of Clusters. Note that eps is affected by hyperspace_distance_percentage. 

### FMI <a name="fmi"></a>
We can see from the graph on the left that as the hyperspace_distance_percentage increases (causing eps to increase), the FMI decreases to a certain value before increasing again. The reason for this pattern quite unintuitive, so lets explore this step by step. We start with the lower end of the X-axis where the eps value is small, very few clusters are formed at this eps value because the model is being very strict in identifying clusters. As majority of the images do not belong in clusters (true labels), having only a few identified clusters results in a generally higher FMI. 

As eps increases, more clusters are identified, but that increases the number of images wrongly clustered as well. Due to how FMI is calculated, having some wrongly predicted images punishes the FMI score heavily, resulting in the drop of FMI. Towards the upper end of the X-axis where eps value is large, the model is lenient in clustering images together, this leads to the model clumping many images together in a single cluster even if the images are not similar looking. Again, due to how FMI is calculated, the FMI score is relatively high even though this (having few gigantic clusters) is not the ideal result.

### Number Of Clusters <a name="number-of-clusters"></a>
Now we look at the relationship between number of clusters and eps (graph on the right). As mentioned in the paragraph above, when eps is small, there are very few number of clusters as the model is strict. As eps increases, the number of clusters increase. However, past a certain point, the number of clusters will start to decrease as the model starts to become too lenient, clumping many images together to form few huge clusters.

### Conclusion <a name="macro-metrics-conclusion"></a>
The conclusion to be drawn from studying these graphs is that FMI score should be taken with a pinch of salt. In reality, FMI score is better suited for use in situations where there are much fewer clusters to be formed, like a business profiling its customers into 3~7 clusters. Whereas in our situation, we have 73 clusters, hence poor FMI scores are to be expected.

Another conclusion to be drawn is that instead of using FMI, we should instead pay attention to number of clusters to be used a rough metric for whether the set of parameters we used were appropriate. That being said, using number of clusters as a metric has its limitations - there might be many clusters formed, but unless we investigate each cluster individually, we cannot gurantee that these are good quality clusters being formed. Another limitation is that when this is used in production, we will not have the true number of clusters of any dataset. For that matter, a dataset might not even have any similar looking images, which will result in number of clusters formed to be zero.



## Visual Inspection Of Different Parameter Combinations <a name="visual-inspection-of-different-parameter-combinations"></a>
Now that we understand the big picture, we can zoom in to view the results of using different combinations of parameters. The goal here is to narrow down the range of parameters to something that produces decent results. Granted this may just be specific to this dataset, but since we're working with percentages here, it will still help us in gaining some intuition as to what is a good range of parameters to use for other datasets. 

We first fix the explained_var to 0.98 and try out different hyperspace_distance_percentage values ranging from 5% to 15%. We found hyperspace_distance_percentage = 10% to be the most optimal value, hence we fixed the hyperspace_distance_percentage to 10% and varied the explained_var from 0.9 to 0.8. Key findings from the various combinations of parameters are listed below. Again, there are **191 images** that belong to one of **73** true clusters identified.


### explained_var 0.98, hyperspace_distance_percentage 10% <a name="ev0.98-hdp0.1"></a>
<p align="center">
    <img width="750" alt="breaks-up-lenient-cluster" src="https://user-images.githubusercontent.com/56946413/129154174-3a4aba35-c3e9-4e42-b8fc-fdaf580c2a24.png">
</p>
<p align="center">Model splits a lenient true cluster into smaller and more accurate clusters</p>

- hyperspace diameter = 402.7 | FMI = 0.493 | Number of predicted images in clusters = 160 | Number of images missing from true clusters = 38
- model is strict (in a good way), splits a lenient true cluster into smaller and more accurate ones (refer to image above)
- very few (only 2) made up clusters


### explained_var 0.98, hyperspace_distance_percentage 5% <a name="ev0.98-hdp0.05"></a>
<p align="center">
    <img width="750" alt="miss-out-obvious-cluster" src="https://user-images.githubusercontent.com/56946413/129155681-317259c9-590c-4154-ae82-299e465badfc.png">
</p>
<p align="center">Model misses out obvious images belonging to clusters</p>

- hyperspace diameter = 402.7 | FMI = 0.626 | Number of predicted images in clusters = **51** | Number of images missing from true clusters = **140**
- although FMI is higher, the model is too strict, misses out on many obvious images that belong to clusters
- only predicts 51 images out of 191 to belong to clusters


### explained_var 0.98, hyperspace_distance_percentage 15% <a name="ev0.98-hdp0.15"></a>
- hyperspace diameter = 402.7 | FMI = 0.427 | Number of predicted images in clusters = **352** | Number of images missing from true clusters = 6
- by increasing hyperspace_distance_percentage to 15%, the model becomes extremely lenient in clustering images, one of the clusters has 285 images
- this is due to how DBSCAN clusters points of data under the hood, increasing eps value to a drastic value causes unwanted clusters to expand rapidly
- only produced 26 decent clusters when there's supposed to be 73


### explained_var 0.9, hyperspace_distance_percentage 10% <a name="ev0.9-hdp0.1"></a>
<p align="center">
    <img width="750" alt="group-2-into-1" src="https://user-images.githubusercontent.com/56946413/129158423-dfdb04e3-ba7b-4e1a-8733-f89e0a28a98e.png">
</p>
<p align="center">Model groups 2 true clusters to form 1 predicted cluster</p>

<p align="center">
    <img width="750" alt="cluster-selfies-together" src="https://user-images.githubusercontent.com/56946413/129158069-837fb499-5cbf-4a4f-ba4a-2cdca6ed8723.png">
</p>
<p align="center">Model groups images of close up faces together</p>

- hyperspace diameter = 401.94 | FMI = 0.438 | Number of predicted images in clusters = 217 | Number of images missing from true clusters = 16
- performance of model is comparable to when explained_var = 0.98, except that this model is slightly more lenient in grouping images together. Refer to the image above where the model grouped together 2 true clusters to form 1 predicted cluster
- with the drop in explained_var from 0.98 to 0.9, it seems that the model loses the ability to detect details and instead groups images together based on shape of objects in the images. This can be seen from the image above where the model clusters several different looking images of close up faces together when in reality, none of these images belong to any true clusters


### explained_var 0.8, hyperspace_distance_percentage 10% <a name="ev0.8-hdp0.1"></a>
- hyperspace diameter = 400.04 | FMI = 0.386 | Number of predicted images in clusters = **269** | Number of images missing from true clusters = 9
- with such a low explained_var, the performance of the model kind of falls off the rails
- the model groups together very different looking images, resulting in a lot of predicted clusters that do not exist in true


### Conclusion <a name="diff-parameter-combination-conclusion"></a>
After testing out 5 different combinations of parameters, the conclusion that can be drawn is that explained_var = 0.98 and hyperspace_distance_percentage = 10% yielded the best results (at least for this dataset). Generally, you would want to keep explained_var high (> 0.95) as decreasing explained_var presumably leads to less amount of details being fed into the model, causing the poor clustering performance of the model. 

We also found that hyperspace_distance_percentage was quite a sensitive parameter as tweaking it by just a few percentage points caused drastic changes in output of the model. Generally, keeping hyperspace_distance_percentage to < 10% should yield promising results. I would recommended starting at 5% and increasing/decreasing accordingly based on results you get from the model.
