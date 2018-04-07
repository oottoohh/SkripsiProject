# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://doc.scrapy.org/en/latest/topics/items.html

import scrapy


class ProjectscrapyItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    SkripsiTitle = scrapy.Field()
    SkripsiAuthor = scrapy.Field()
    # SkripsiAdvisor = scrapy.Field()
    # SkripsiAdvisor1 = scrapy.Field()
    SkripsibyKeyword = scrapy.Field()    
    SkripsibyDate = scrapy.Field()
    # SkripsiPublisher = scrapy.Field()
    # SkripsiSeries = scrapy.Field()
    SkripsiAbstrak = scrapy.Field()
    # SkripsiDeskripsi = scrapy.Field()
    SkripsiURI = scrapy.Field()
    # SkripsiCollection = scrapy.Field()
