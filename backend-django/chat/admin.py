from django.contrib import admin  # type: ignore

from chat.models import BERT, T5Predictor, TFIDF, Word2VecModel

# Register your models here.
admin.site.register(BERT)
admin.site.register(T5Predictor)
admin.site.register(TFIDF)
admin.site.register(Word2VecModel)
