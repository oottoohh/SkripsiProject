import scrapy
from ProjectScrapy.items import ProjectscrapyItem
from datetime import datetime
import re


class data_scrape(scrapy.Spider):
	name = "dataScraper"

	# First Start Url
	start_urls = ['http://repository.uinjkt.ac.id/dspace/handle/123456789/160/browse?type=dateaccessioned&sort_by=2&order=DESC&rpp=480&etal=7&submit_browse=Update']

	def parse(self, response):
		for href in response.xpath("//tr/td[@headers='t2']/a/@href"):
			# add the scheme, eg http://
			url  = "http://repository.uinjkt.ac.id" + href.extract() 
			yield scrapy.Request(url, callback=self.parse_dir_contents)	
					
	def parse_dir_contents(self, response):
		item = ProjectscrapyItem()
		
		title  =  response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()
		Author =  response.xpath("//tr/td[@class='metadataFieldValue']/a[@class='author']/text()").extract()
		SkripsibyKeyword = response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()
		SkripsibyDate = response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()
		SkripsiAbstrak = response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()
		SkripsiURI = response.xpath("//tr/td[@class='metadataFieldValue']/a/@href").extract()
		try:
			item['SkripsiTitle'] = title[0]
		except IndexError:
			item['SkripsiTitle'] =''
		try:	
			item['SkripsiAuthor']= Author
		except IndexError:
			item['SkripsiAuthor']= ''

		# item['SkripsiAdvisor'] = response.xpath("//tr/td[@class='metadataFieldValue']/a[@class='author']/text()").extract()[1]

		# item['SkripsiAdvisor1'] = response.xpath("//tr/td[@class='metadataFieldValue']/a[@class='author']/text()").extract()[2]

		try:	
			item['SkripsibyKeyword']= SkripsibyKeyword[1]
		except IndexError:
			item['SkripsibyKeyword']= ''
		try:
			item['SkripsibyDate'] = SkripsibyDate[2]
		except IndexError:
			item['SkripsibyDate'] =''
		try:	
			item['SkripsiAbstrak']= SkripsiAbstrak[5]
		except IndexError:
			item['SkripsiAbstrak']= ''
		try:	
			item['SkripsiURI']= SkripsiURI
		except IndexError:
			item['SkripsiURI']= ''

		# item['SkripsibyDate'] =  response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()[2]
		
		# item['SkripsiPublisher'] = response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()[2]

		# item['SkripsiSeries'] =  response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()[3]

		# item['SkripsiAbstrak'] =  response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()[5]
		
		# item['SkripsiDeskripsi'] =  response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()[5]
		
		# item['SkripsiURI'] =  response.xpath("//tr/td[@class='metadataFieldValue']/a/@href").extract()[4]
	
		# item['SkripsiCollection'] =  response.xpath("//tr/td[@class='metadataFieldValue']/text()").extract()[6]
		

		return item

