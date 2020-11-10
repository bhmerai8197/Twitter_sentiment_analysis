from django.http import HttpResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

def index(request):
    if request.method == 'POST':
        search_data : request.POST.get('twitter_search')

        contex = {
            'twitter_search': search_data,
        }

        template =loader.get_template('search.html')
        return HttpResponse(template.render(contex, request))
    else:
        template = loader.get_template('index.html')
        return HttpResponse(template.render(template.render()))