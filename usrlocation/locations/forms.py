from django import forms

class EmailForm(forms.Form):
    email_id = forms.EmailField(label='email ID', max_length=100)
