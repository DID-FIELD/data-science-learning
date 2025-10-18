#!PYTHONPATH (sys.executable)
import time
import pymysql
html5top='''
<!-- {fname} -->
<!DOCTYPE html>
<html>
 <head>
  <title>{title}</title>
 </head>
 <body>
  <h1>{header}</h1>
'''
html5bottom='''
 </body>
</html>
'''
tableHeader='''
<table border=1>
<tr><th>customerName</th><th>phone</th><th>country</th></tr>
'''
print (html5top.format(fname='pydb.py',title='pymysql',header='Customer Table'))

connection = pymysql.connect(user='root',
                             password='',
                             db='classicmodels', 
                             cursorclass=pymysql.cursors.DictCursor)
print(tableHeader)
with connection.cursor() as cursor:
    cursor.execute('SELECT customerName,phone,country from customers')
    for e in cursor.fetchall():
        print ('<tr><td>'+e['customerName']+'</td>'+
               '<td>'+e['phone']+'</td>'+
               '<td>'+e['country']+'</td></tr>')

print('</table>')
connection.close()
print (time.ctime( time.time() ))
print (html5bottom)