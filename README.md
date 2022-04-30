# image-deduplicate

## Table Of Contents
1. [Context](#context)
    1. [Capture and Tag](#captureandtag)
    2. [Purpose](#purpose)
    3. [Limitation](#limitation)
3. [Research](#research)
4. [Application](#application)
5. [Misc](#misc)

## Context <a name="context"></a>
### Capture and Tag <a name="captureandtag"></a>
To set the stage as to what the purpose of this use case is for, we first have to understand what **Capture and Tag** is. Capture and Tag is a product that the company intends to roll out in the future. The goal of Capture and Tag is to use crowdsourcing as a means to collect specific data requested by clients. 

For example, Tesla is soon reaching Singapore's shores and they might require images of traffic lights in Singapore to train their full-self-driving software to be contextualized to Singapore's traffic situation. Having Capture and Tag would enable the company to utilize its tagger pool to collect images of traffic lights in a quick and efficient manner.

Taggers will then be compensated accordingly based on the number images that they have submitted. However, how does the company determine how much remuneration is to be correctly compensated to each tagger? For example, whats to stop a tagger from submitting several duplicates of an image, or even taking 100 pictures of the same traffic light but only changing the angle taken by only 1 degree each time?

This is where the image-deduplication use case comes in.

### Purpose <a name="purpose"></a>
The goal of this use case is to build an algorithm that is able to *look* through a dataset of images and identify similar looking images, aka image-deduplication. The user of this system (could be either the company's Ops team or the client) will be able to choose how strict the model is in identifying similar looking images via changing arguments accepted by the model. This way, they have a means to efficiently filter out similar looking images in the dataset instead of having to manually go through each photo one by one. 

### Limitation <a name="limitation"></a>
One limitation identified is the lack of data available for testing the model. As will be explained later, there were three datasets that were used when testing the image-deduplication algorithm, Caltech101, iPhone gallery images and beverage can images submission. The model has performed decently on these datasets, but testing with a greater variety of datasets will be needed to gain better intuition as to what the optimal range of the arguments is. 


## Research <a name="research"></a>
The research folder contains a python notebook that details every step of the image-deduplication algorithm process. The folder also contains markdown files that note down the research conducted thus far.

## Application <a name="application"></a>
The application folder contains python scripts that run a local Flask server that contains the packaged up image deduplication API. The files in the application folder are the end product of the work conducted in the research folder.
