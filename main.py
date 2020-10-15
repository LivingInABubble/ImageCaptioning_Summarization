import argparse
import json
import os
from sys import stdout
from time import time

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
# from matplotlib.pyplot import imread
from imageio import imread
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from rouge_score import rouge_scorer
from summarizer import Summarizer, TransformerSummarizer
from tqdm import tqdm
# from tqdm.notebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=3):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model
    :param decoder: decoder model
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    try:
        img = imread(image_path)
    except:
        print('wrong image format')
        return False

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)

    # img = imresize(img, (256, 256))
    img = np.array(Image.fromarray(img).resize((256, 256)))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])

    try:
        image = transform(img)  # (3, 256, 256)
    except:
        print('unsupported image')
        return False

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:
        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)
        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)
        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)
        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe
        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)
        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)
        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = top_k_words // vocab_size  # (s)
        next_word_inds = top_k_words % vocab_size  # (s)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq = complete_seqs[i]
    # alphas = complete_seqs_alpha[i]

    return seq  # , alphas


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: caption
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()


def get_text(image_id):
    image_id = os.path.split(image_id)[-1].split('.')[0]
    with open('ManyModalQAData/official_aaai_split_train_data.json') as f:
        train_data = json.load(f)

    for element in train_data:
        if image_id == element['id']:
            return sent_tokenize(element['text'].strip())
    else:
        with open('ManyModalQAData/official_aaai_split_dev_data.json') as f:
            dev_data = json.load(f)

        for element in dev_data:
            if image_id == element['id']:
                return sent_tokenize(element['text'].strip())
        else:
            return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--img', '-i', help='path to image folder', default='ManyModalImages')
    parser.add_argument('--ckpt', '-c', help='path to multimodal model checkpoint',
                        default='BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')
    parser.add_argument('--word_map', '-wm', help='path to word map JSON',
                        default='WORDMAP_coco_5_cap_per_img_5_min_word_freq.json')
    parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    parser.add_argument('--model', '-m', help='model for summarization', default='bert')

    args = parser.parse_args()

    # Load model
    checkpoint = torch.load(args.ckpt, map_location=str(device))
    decoder = checkpoint['decoder']
    decoder = decoder.to(device)
    decoder.eval()
    encoder = checkpoint['encoder']
    encoder = encoder.to(device)
    encoder.eval()

    if args.model == 'bert':
        summarizer = Summarizer()
    elif args.model == 'gpt':
        summarizer = TransformerSummarizer(transformer_type='GPT2', transformer_model_key='gpt2')
    else:
        print('select bert or gpt')
        return

    # Load word map (word2ix)
    with open(args.word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    with open('rough score.csv', 'w') as file:
        file.writelines('filename,precision,recall,f1measure\n')

    for image in tqdm(sorted(os.listdir(args.img)), file=stdout):
        print(image)

        t = time()
        img_path = os.path.join(args.img, image)
        # Encode, decode with attention and beam search
        seq = caption_image_beam_search(encoder, decoder, img_path, word_map, args.beam_size)
        if not seq:
            continue

        # alphas = torch.FloatTensor(alphas)

        # Visualize caption and attention of best sequence
        # visualize_att(img_path, seq, alphas, rev_word_map, args.smooth)
        print('time for image captioning:', time() - t)

        keywords = [rev_word_map[s] for s in seq]
        stop_words = set(stopwords.words('english'))
        keywords = [keyword for keyword in list(set(keywords[1:-1])) if keyword not in stop_words]
        # print(keywords)

        texts = get_text(image)
        sentences = []
        for text in texts:
            for keyword in keywords:
                if keyword in text:
                    sentences.append(text)
                    break

        texts, sentences = ' '.join(texts), ' '.join(sentences)

        t = time()
        summary1 = summarizer(texts)
        print('time for summarization on whole texts:', time() - t)

        t = time()
        summary2 = summarizer(sentences)
        print('time for summarization on selected sentences:', time() - t)

        t = time()
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
        scores = scorer.score(summary1, summary2)['rouge1']
        print('rouge1:', scores)
        with open('rough score.csv', 'a') as file:
            file.write(','.join([image, str(scores.precision), str(scores.recall), str(scores.fmeasure), '\n']))

        print('time for calculating rouge score:', time() - t)


if __name__ == '__main__':
    main()
