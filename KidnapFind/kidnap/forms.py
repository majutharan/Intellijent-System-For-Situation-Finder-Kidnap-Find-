from django import forms


class VideoForm(forms.Form):
    uid = forms.CharField(max_length=100)
    video = forms.FileField()


class AudioForm(forms.Form):
    uid = forms.CharField(max_length=100)
    audio = forms.FileField()


class RegisterForm(forms.Form):
    id = forms.CharField(max_length=100)
    name = forms.CharField(max_length=100)
    age = forms.CharField(max_length=100)
    place = forms.CharField(max_length=100)
    work = forms.CharField(max_length=100)
    sex = forms.CharField(max_length=100)
    mail_id = forms.CharField(max_length=100)


class IdForm(forms.Form):
    id = forms.CharField(max_length=100)




