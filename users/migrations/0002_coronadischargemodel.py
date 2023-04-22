# Generated by Django 2.0.13 on 2020-08-28 05:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='CoronaDischargeModel',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('username', models.CharField(max_length=100)),
                ('email', models.CharField(max_length=100)),
                ('loginid', models.CharField(max_length=100)),
                ('filename', models.CharField(max_length=100)),
                ('file', models.FileField(upload_to='files/')),
                ('redColor', models.FloatField()),
                ('greenColor', models.FloatField()),
                ('blueColor', models.FloatField()),
                ('picHeight', models.FloatField()),
                ('picWidht', models.FloatField()),
                ('blockofPixel', models.FloatField()),
                ('picbrightness', models.FloatField()),
                ('cdate', models.DateTimeField(auto_now_add=True)),
            ],
            options={
                'db_table': 'CoronaDischargeImages',
            },
        ),
    ]
