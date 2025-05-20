import numpy as np
#import numpy


def i2t_SCAN(sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    #当前样本的index找到对应的相似度向量：当前图像与所有文本的相似度，与当前图像匹配的文本相似度在向量的5* index, 5 * index+5的部分
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1] #得到相似度降序之后样本索引。[a,b,c,d,e.....]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1): #正样本的索引
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank #存每个图像的五个文本的最佳排名故ranks长度为5000（个图像）
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(np.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr)

def t2i_SCAN(sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)#存储每个文本的正样本图像排名
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(np.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr)

'''def i2t(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    index_list = []

    ranks = numpy.zeros(npts)
    top1 = numpy.zeros(npts)
    for index in range(npts):

        # Get query image
        im = images[5 * index].reshape(1, images.shape[1])

        # Compute scores
        d = numpy.dot(im, captions.T).flatten()
        inds = numpy.argsort(d)[::-1]
        index_list.append(inds[0])

        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = numpy.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)

    r20 = 100.0 * len(numpy.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(numpy.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(numpy.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(numpy.where(ranks < 100)[0]) / len(ranks)

    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100,  medr, meanr)


def t2i(images, captions, npts=None, measure='cosine', return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    if npts is None:
        npts = int(images.shape[0] / 5)
    ims = numpy.array([images[i] for i in range(0, len(images), 5)])

    ranks = numpy.zeros(5 * npts)
    top1 = numpy.zeros(5 * npts)
    for index in range(npts):

        # Get query captions
        queries = captions[5 * index:5 * index + 5]

        # Compute scores
       
        d = numpy.dot(queries, ims.T)
        inds = numpy.zeros(d.shape)
        for i in range(len(inds)):
            inds[i] = numpy.argsort(d[i])[::-1]
            ranks[5 * index + i] = numpy.where(inds[i] == index)[0][0]
            top1[5 * index + i] = inds[i][0]

    # Compute metrics
    r1 = 100.0 * len(numpy.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(numpy.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(numpy.where(ranks < 10)[0]) / len(ranks)

    r20 = 100.0 * len(numpy.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(numpy.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(numpy.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(numpy.where(ranks < 100)[0]) / len(ranks)

    medr = numpy.floor(numpy.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr)'''
def i2t_SCAN_NN(sims, return_ranks=False):
    """
    Images->Text (Image Annotation)
    sims: (N, N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 对每个图像与文本的相似度从大到小排序，返回排序后的索引
        rank = np.where(inds == index)[0][0]  # 找到对应文本描述在排序后的索引中的位置
        ranks[index] = rank  # 存储当前图像的最佳排名
        top1[index] = inds[0]  # 存储最相似的文本描述的索引

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(np.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr)

def t2i_SCAN_NN(sims, return_ranks=False):
    """
    Text->Images (Text Annotation)
    sims: (N, N) matrix of similarity text-image
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    sims=sims.T

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]  # 对每个文本与所有图像的相似度进行降序排序，并返回排序后的索引
        rank = np.where(inds == index)[0][0]  # 找到对应图像在排序后的索引中的位置
        ranks[index] = rank  # 存储当前文本的最佳排名
        top1[index] = inds[0]  # 存储最相似的图像的索引

    # 计算指标
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r20 = 100.0 * len(np.where(ranks < 20)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r70 = 100.0 * len(np.where(ranks < 70)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)

    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r20, r50, r70, r100, medr, meanr)