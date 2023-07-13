"""
Implements class used to visualize experiment results in an HTML report.
"""

#  Example on how to use Html class:
#    html = Html('index.html')
#    html.head('Hi!')
#    html.open_table(['Name','Birth date'])
#    html.add_table_row(['Peter','1.3.1900'])
#    html.add_table_row(['Thomas','2.4.1905'])
#    html.close_table()
#    html.add_img('img.png',width=200,height=200,alt='JOE')
#    df = pd.DataFrame(data={'col1': [1, 2], 'col2': [3, 4]})
#    html.add_string(df.to_html())
#    html.tail()
#    html.close()


class Html:
    def __init__(self, file_name):
        self.file = open(file_name, 'w+')

    def head(self, title):
        self.file.write(
            '<!doctype html public "-//w3c//dtd html 4.0 transitional//en">\n')
        self.file.write('<html>\n')
        self.file.write('<head>\n')
        self.file.write(f'<title> {title} </title>\n')
        self.file.write('</head>\n')
        self.file.write(
            '<body text="#000000" bgcolor="#FFFFFF" link="#0000EF" vlink="#51188E" alink="#FF0000">\n')

    def header1(self, text):
        self.file.write(f'<h1>{text}</h1>\n')

    def header2(self, text):
        self.file.write(f'<h2>{text}</h2>\n')

    def header3(self, text):
        self.file.write(f'<h3>{text}</h3>\n')

    def add_paragraph(self, text):
        self.file.write(f'<p>\n')
        self.file.write(f'{text}')
        self.file.write(f'</p>\n')

    def open_table(self, headers=None, border=0):
        self.file.write(f'<table border="{border}">\n')
        if headers is not None:
            self.file.write('<tr>')
            for head in headers:
                self.file.write(f'<th>{head}</th>')
            self.file.write('</tr>\n')

    def add_table_row(self, columns):
        self.file.write('<tr>\n')
        for col in columns:
            self.file.write(f'<td>{col}</td>')
        self.file.write('</tr>\n')

    def add_img(self, image_file, width=None, height=None, alt=None):
        txt = f'<img src={image_file} '
        if width is not None:
            txt = txt + f'width="{width}"'
        if height is not None:
            txt = txt + f'height="{height}"'
        if alt is not None:
            txt = txt + f'alt="{alt}"'
        txt = txt + '>'
        self.file.write(f'{txt}\n')

    def close_table(self):
        self.file.write('</table>\n')

    def tail(self):
        self.file.write('</body>\n')
        self.file.write('</html>\n')

    def close(self):
        self.file.close()

    def add_string(self, str):
        self.file.write(str)
