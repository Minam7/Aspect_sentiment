# !/usr/bin/python
# -*- coding: utf-8 -*-

import codecs

'''
deleted aspect:

'مواد': ['پلاستیک', 'پلاستیکی', 'فلز', 'فلزی', 'فلزیه', 'آهن', 'آهنیه', 'آهنی', 'استیل', 'طلا', 'طلاس',
                    'طلاست', 'نقره', 'نقرس', 'نقرست', 'آلومینیوم', 'آلومینیومی', 'فولاد', 'فولادی'],

'''

aspects = {'کیفیت': ['کیفیت', 'کیفیتش', 'کیفیته', 'ساخت'],
           'قیمت': ['قیمت', 'قیمتش', 'قیمته', 'قیمتا', 'قیمتی', 'قیمتاش', 'قیمتشون', 'ارزون', 'ارزان', 'گرون', 'گران',
                    'گرانتر', 'گرانترین', 'گران‌تر', 'گران‌ترین', 'گرونتر', 'گرونترین', 'گرون‌تر', 'گرون‌ترین',
                    'گرانتره', 'گرانترینه', 'گران‌تره', 'گران‌ترینه', 'گرونتره', 'گرونترینه', 'گرون‌تره', 'گرون‌ترینه',
                    'ارزانتر', 'ارزانترین', 'ارزان‌تر', 'ارزان‌ترین', 'ارزانتر', 'ارزانترین', 'ارزان‌تر', 'ارزان‌ترین',
                    'ارزانتره', 'ارزانترینه', 'ارزان‌تره', 'ارزان‌ترینه', 'ارزانتره', 'ارزانترینه', 'ارزان‌تره',
                    'ارزان‌ترینه'],
           'طراحی': ['طراحی', 'طراحیش', 'طراح', 'طرح', 'طرحش', 'شکل', 'شکلش'],
           'برند': ['برند', 'برندش', 'برنده'],
           'دوربین': ['دوربین', 'دوربینش', 'دوربینه', 'عکس', 'عکسش', 'عکساش', 'عکاسی', 'عکاسیش'],
           'راحتی': ['راحتی', 'راحت', 'راحته', 'راحتیش'],
           'جنس': ['جنس', 'جنسشون', 'جنسش'],
           'زیبایی': ['زیبا', 'زیباس', 'زیبایی', 'زیباست', 'قشنگ', 'قشنگه', 'خوشگل', 'خوشگله', 'شیک', 'شیکه', 'ظرافتش',
                      'ظرافت', 'ظریفه', 'ظریف'],
           'رنگ': ['رنگ', 'رنگش'],
           'کارایی': ['کار', 'کارا', 'کارایی', 'کاربرد', 'کاربردی', 'استفاده', 'استفادش', 'اسفاده'],
           'صدا': ['صدا', 'صداش'],
           'شارژ': ['شارژ', 'شارژر', 'شارژش', 'شارژرش', 'باتری', 'باتریش', 'باطری', 'باطریش'],
           'اصالت': ['اصل', 'اصله', 'اصالت', 'اورجینال', 'اورجیناله'],
           'صفحه': ['صفحه', 'نمایش', 'نمایشش', 'اسکرین', 'اسکرینش', 'صفحش'],
           'ابعاد': ['ارتفاع', 'ارتفاعش', 'اندازه', 'اندازش', 'اندازشون', 'قد', 'قدش', 'طول', 'طولش', 'عرض', 'عرضش',
                     'عمق', 'عمقش', 'ظرفیت', 'ظرفیتش'],
           'وزن': ['وزن', 'وزنش', 'سبک', 'سنگین', 'سبکه', 'سنگینه'],
           'بدنه': ['بدنه', 'بدنش', 'اسکلت', 'قالب', 'اسکلتش', 'قالبش'],
           'دقت': ['دقت', 'دقتش', 'دقیق', 'دقیقه'],
           'شبکه': ['اینترنت', 'نت', 'شبکه', 'اینترنتش', 'آنتن', 'آنتنش', 'نتش', 'شبکش'],
           'حفاظت': ['حفاظت', 'محافظت', 'محافظ', 'محافظش', 'مراقبت', 'مراقب', 'مراقبش'],
           'بسته‌بندی': ['جعبه', 'بسته', 'کارتن', 'بستش', 'جعبش', 'کارتنش']}


def load_embedding():
    word_dict = dict()
    embedding = list()

    f_vec = codecs.open('fastText/wiki.fa.vec', 'r', 'utf-8')

    idx = 0
    for line in f_vec:
        if len(line) < 100:
            continue
        else:
            component = line.strip().split(' ')
            word_dict[component[0].lower()] = idx
            word_vec = list()
            for i in range(1, len(component)):
                word_vec.append(float(component[i]))
            embedding.append(word_vec)
        idx = idx + 1
    f_vec.close()
    word_dict['<padding>'] = idx
    embedding.append([0.] * len(embedding[0]))
    word_dict_rev = {v: k for k, v in word_dict.items()}
    return word_dict, word_dict_rev, embedding


def load_stop_words():
    stop_words = list()
    fsw = codecs.open('data/dictionary/stop_words.txt', 'r', 'utf-8')
    for line in fsw:
        stop_words.append(line.strip())
    fsw.close()
    return stop_words


def load_sentiment_dictionary():
    pos_list = list()
    neg_list = list()
    rev_list = list()
    inc_list = list()
    dec_list = list()
    sent_words_dict = dict()

    fneg = open('data/dictionary/negative_words.txt', 'r')
    fpos = open('data/dictionary/positive_words.txt', 'r')
    frev = open('data/dictionary/reverse_words.txt', 'r')
    fdec = open('data/dictionary/decremental_words.txt', 'r')
    finc = open('data/dictionary/incremental_words.txt', 'r')

    for line in fpos:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 0
            pos_list.append(line.strip())

    for line in fneg:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 1
            neg_list.append(line.strip())

    for line in frev:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 2
            rev_list.append(line.strip())

    for line in finc:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 3
            inc_list.append(line.strip())

    for line in fdec:
        if not line.strip() in sent_words_dict:
            sent_words_dict[line.strip()] = 4
            dec_list.append(line.strip())

    fneg.close()
    fpos.close()
    frev.close()
    fdec.close()
    finc.close()

    return pos_list, neg_list, rev_list, inc_list, dec_list, sent_words_dict


def export_aspect():
    return set(aspects.keys())


def load_data(label_dict, seq_max_len, negative_weight, positive_weight):
    train_data = list()
    train_mask = list()
    train_binary_mask = list()
    train_label = list()
    train_seq_len = list()
    train_sentiment_for_word = list()
    count_pos = 0
    count_neg = 0

    stop_words = load_stop_words()
    pos_list, neg_list, rev_list, inc_list, dec_list, sent_words_dict = load_sentiment_dictionary()
    aspect_list = export_aspect()
    word_dict, word_dict_rev, embedding = load_embedding()

    # load data, mask, label

    f_processed = codecs.open('data/all_comments.txt', 'r', 'utf-8')
    for line in f_processed:
        data_tmp = list()
        mask_tmp = list()
        binary_mask_tmp = list()
        label_tmp = list()
        sentiment_for_word_tmp = list()
        count_len = 0

        words = line.strip().split(' ')
        for word in words:
            if word in stop_words:
                continue
            word_clean = word.replace('{aspositive}', '').replace('{asnegative}', '')

            if word_clean in word_dict.keys() and count_len < seq_max_len:
                if word_clean in pos_list:
                    sentiment_for_word_tmp.append(1)
                elif word_clean in neg_list:
                    sentiment_for_word_tmp.append(2)
                elif word_clean in rev_list:
                    sentiment_for_word_tmp.append(0)
                elif word_clean in inc_list:
                    sentiment_for_word_tmp.append(0)
                elif word_clean in dec_list:
                    sentiment_for_word_tmp.append(0)
                else:
                    sentiment_for_word_tmp.append(0)

                if 'aspositive' in word:
                    mask_tmp.append(positive_weight)
                    binary_mask_tmp.append(1.0)
                    label_tmp.append(label_dict['aspositive'])
                    count_pos = count_pos + 1
                elif 'asnegative' in word:
                    mask_tmp.append(negative_weight)
                    binary_mask_tmp.append(1.0)
                    label_tmp.append(label_dict['asnegative'])
                    count_neg = count_neg + 1
                else:
                    mask_tmp.append(0.)
                    binary_mask_tmp.append(0.)
                    label_tmp.append(0)
                count_len = count_len + 1

                data_tmp.append(word_dict[word_clean])

        train_seq_len.append(count_len)

        for _ in range(seq_max_len - count_len):
            data_tmp.append(word_dict['<padding>'])
            mask_tmp.append(0.)
            binary_mask_tmp.append(0.)
            label_tmp.append(0)
            sentiment_for_word_tmp.append(0)

        train_data.append(data_tmp)
        train_mask.append(mask_tmp)
        train_binary_mask.append(binary_mask_tmp)
        train_label.append(label_tmp)
        train_sentiment_for_word.append(sentiment_for_word_tmp)

    f_processed.close()

    print('pos: %d' % count_pos)
    print('neg: %d' % count_neg)
    print('len of train data is %d' % (len(train_data)))
    data_sample = ''
    for id in train_data[10]:
        data_sample = data_sample + ' ' + word_dict_rev[id]

    print('%s' % data_sample)
    print(train_data[10])
    print(train_mask[10])
    print(train_label[10])
    print(train_sentiment_for_word[10])
    print('len of word dictionary is %d' % (len(word_dict)))
    print('len of embedding is %d' % (len(embedding)))
    print('len of aspect_list is %d' % (len(aspect_list)))

    return train_data, train_mask, train_binary_mask, train_label, train_seq_len, train_sentiment_for_word, \
           word_dict, word_dict_rev, embedding, aspect_list


def main():
    seq_max_len = 120
    negative_weight = 2.5
    positive_weight = 1.0

    label_dict = {
        'aspositive': 1,
        'asnegative': 2
    }

    train_data, train_mask, train_binary_mask, train_label, train_seq_len, train_sentiment_for_word, word_dict, \
    word_dict_rev, embedding, aspect_list = load_data(
        label_dict,
        seq_max_len,
        negative_weight,
        positive_weight,
    )


if __name__ == '__main__':
    main()
