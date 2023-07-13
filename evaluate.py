import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import yaml
import os
import torch
from lib.utils import create_dir
from lib.utils import get_loss_matrix
from lib.evaluation import *
from lib.html import Html
from lib.data_loaders import load_face_list, MyYamlLoader

def print_benchmark(html, benchmark ):
    """Print databases and trn/val/splist defining the benchmark.
    
    Args:
        html (HTML class): output HTML file.
        benchmark (dict): content of benchmark YAML.
    Returns:
    """
    for db in benchmark:
        html.header2( f"{db['tag']}")
        html.add_paragraph(f"file: {db['database']}")
        html.open_table( [""] + [f"split {i}" for i in range(len(db['split']))], border=1 )
        for part in ['trn','val','tst']:
            str = [part]
            for i, split in enumerate( db['split'] ):
                str.append( f"{split[part]}" )
            html.add_table_row( str )
        html.close_table()

def print_config( html, config):
    """Print config file.
    Args:
        html (HTML): output HTML file.
        config (dict): content of config YAML.
    Returns:
    """
    html.header2("Data:")
    html.open_table(border=1)
    for key, value in config['data'].items():
        html.add_table_row( [f'{key}', f'"{value}"'])
    html.close_table()

    html.header2("Heads:")
    for i, head in enumerate(config['heads']):
        html.header3(f"({i}) {head['tag']}")
        html.open_table( border=1 )
        for key, value in head.items():
            if key != 'tag':
                html.add_table_row( [f"{key}", f"{value}"])
        html.close_table()

    html.header2("Model:")
    html.open_table(border=1)
    for key, value in config['model'].items():
        html.add_table_row( [f'{key}', f'"{value}"'])
    html.close_table()

    html.header2("Optimizer:")
    html.open_table(border=1)
    for key, value in config['optimizer'].items():
        html.add_table_row( [f'{key}', f'"{value}"'])
    html.close_table()

    html.header2("Preprocess:")
    html.open_table(border=1)
    for key, value in config['preprocess'].items():
        html.add_table_row( [f'{key}', f'"{value}"'])
    html.close_table()


def compute_uncertainty( prediction ):
    prediction['uncertainty'] = {}
    for head in prediction['config']['heads']:
        tag = head['tag']
        loss_matrix = get_loss_matrix( len( head['labels'] ), head['metric'][0] )
        pred_label = prediction['predicted_label'][tag]
        post = prediction['posterior'][tag]

        n = prediction['posterior'][tag].shape[0]
        prediction['uncertainty'][tag] = np.empty( n )
        for i in range(n):
            prediction['uncertainty'][tag][i] = np.dot( loss_matrix[:,pred_label[i]], post[i,:] )

    return


if __name__ == '__main__':

    if len( sys.argv ) < 2 or len( sys.argv ) > 3:
        print(f"usage: {sys.argv[0]} path/config.yaml")
        sys.exit(f"usage: {sys.argv[0]} path/config.yaml tag")

    config_file = sys.argv[1]

    if len( sys.argv) > 2:
        git_versioning = sys.argv[2]
    else:
        git_versioning = ''

    ######################################################################
    # load config, benchmarks and setup basic variables
    ######################################################################
    with open( config_file,'r') as stream:
        config = yaml.load(stream, Loader=MyYamlLoader )

    with open( config["data"]["benchmark"],'r') as stream:
        benchmark = yaml.load(stream, MyYamlLoader )

    n_splits = len( benchmark[0]['split'] )
    config_name = os.path.basename( config_file ).split('.')[0]
    output_dir = config["data"]["output_dir"] + config_name + "/" + git_versioning + "/"
    data_dir = config["data"]["data_dir"] + config_name + "/"
    report_img_dir = output_dir + "figs/"
    html_report_file = output_dir + "evaluation_index.html"

    ######################################################################
    # load predictions
    ######################################################################
    prediction = []
    for split in range(n_splits):
        predictions_fname = output_dir + f"split{split}/evaluation.pt"
        prediction.append( torch.load( predictions_fname) )

        # if they do not exist compute uncertainty as conditional risk
        if 'uncertainty' not in prediction[-1].keys():
            compute_uncertainty( prediction[-1] )

    ######################################################################
    # load face list
    ######################################################################
    face_list = load_face_list( data_dir + 'face_list.csv' )

    #######################################################################
    # Open html file and prepare folder for report images
    #######################################################################
    html = Html( html_report_file )
    html.head( f"{config_name}")
    html.header1( f"Experiment: {config_name}" )
    html.add_paragraph( f'config file: "{config_file}"')
    create_dir( report_img_dir )

    #######################################################################
    # Config and benchmark
    #######################################################################

    html.header1("Config")
    print_config( html, config )
    html.header1("Benchmark")
    print_benchmark( html, benchmark)


    #######################################################################
    # Error summary table visualize it
    #######################################################################
    error_summary = ErrorSummary( config, benchmark, face_list, prediction )
                
    html.header1("Results")
    html.open_table(['Trn','Val','Tst'])
    html.add_table_row([error_summary.table[x].to_html(index=(x=='trn')) for x in ['trn','val','tst']])
    html.close_table()

    #######################################################################
    # Confusion matrices, risk-coverage curve, risk-uncertainty curve, 
    # uncertainty distirbution
    #######################################################################
    visual_matric = VisualMetric(config, benchmark, face_list, prediction)

    uncertainty = Uncertainty( config, benchmark, face_list, prediction )

    for database in visual_matric.databases:
        html.header2(f"[{database}]")
        for head in config['heads']:
            attr = head['tag']
            for i, matric in enumerate(head['visual_metric']):
                # confuction matrix based matric :-)
                visual_matric.plot( database, attr, matric )
                img_path = f"figs/{database}_{i}_{attr}.svg"
                plt.savefig( output_dir + img_path )
                html.add_img(img_path,alt=f"{attr} on {database}")
                img_path = f"figs/{database}_{i}_{attr}.png"
                plt.savefig( output_dir + img_path )
                plt.close()

            # Risk-Converare curve
            uncertainty.plot_rc_curve( database, attr )
            img_path = f"figs/rccurve_{database}_{attr}.svg"
            plt.savefig( output_dir + img_path, bbox_inches='tight' )
            html.add_img(img_path,alt=f"{attr} on {database}")
            img_path = f"figs/rccurve_{database}_{attr}.png"
            plt.savefig( output_dir + img_path )
            plt.close()

            # Risk-Uncertainty curve
            uncertainty.plot_ru_curve( database, attr )
            img_path = f"figs/rucurve_{database}_{attr}.svg"
            plt.savefig( output_dir + img_path, bbox_inches='tight' )
            html.add_img(img_path,alt=f"{attr} on {database}")
            img_path = f"figs/rucurve_{database}_{attr}.png"
            plt.savefig( output_dir + img_path )
            plt.close()

            # Uncertainty distribution
            uncertainty.plot_uncertainty_hist( database, attr )
            img_path = f"figs/unchist_{database}_{attr}.svg"
            plt.savefig( output_dir + img_path, bbox_inches='tight' )
            html.add_img(img_path,alt=f"{attr} on {database}")
            img_path = f"figs/unchist_{database}_{attr}.png"
            plt.savefig( output_dir + img_path )
            plt.close()

    #######################################################################
    # Close HTML
    #######################################################################
    html.tail()
    html.close()
