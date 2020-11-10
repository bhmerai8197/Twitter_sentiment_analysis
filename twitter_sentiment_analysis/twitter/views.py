import json
from django.shortcuts import render
from django.http import HttpResponse
from . import tweet


def index(request):
    return render(request, 'index.html')


def search(request):
    if request.method == 'POST':
        post_text = request.POST.get('twitter_search')
        tweets,common_word,sentiment,Tweet_analysis,cp,cneg,cn = tweet.data(post_text)

    context = {
        'twitter_search': post_text,
        'tweets': tweets,
        'common_word': common_word,
        'sentiment': sentiment,
        'Tweet_analysis': Tweet_analysis,
                'cp': cp,
            'cneg': cneg,
                'cn': cn,


    }
    return render(request, 'search.html', context)

def loadjson(request):
    with open('search.json', 'r') as jsonfile:
        data = json.load(jsonfile)

    return HttpResponse(json.dumps(data), content_type='application/json')


def loadjson_1(request):
    with open('emotion.json', 'r') as jsonfile:
        data = json.load(jsonfile)

    return HttpResponse(json.dumps(data), content_type='application/json')

