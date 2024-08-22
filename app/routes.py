from flask import render_template, redirect, url_for, request, flash, current_app, session
from app.forms import RegistrationForm, LoginForm
from flask import Blueprint
import csv
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter
import numpy as np

main = Blueprint('main', __name__)

# Helper function to check if user is logged in
def login_required(f):
    def wrap(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('main.login'))
        return f(*args, **kwargs)
    wrap.__name__ = f.__name__
    return wrap

@main.route('/')
def home():
    return render_template('home.html')

@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        supabase = current_app.config['SUPABASE_CLIENT']
        user = supabase.table('users').insert({
            "name": form.name.data,
            "email": form.email.data,
            "phone": form.phone.data,
            "password": form.password.data
        }).execute()
        flash('Account created successfully!', 'success')
        return redirect(url_for('main.login'))
    return render_template('register.html', form=form)

@main.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        supabase = current_app.config['SUPABASE_CLIENT']
        response = supabase.table('users').select("*").eq('email', form.email.data).single().execute()
        user_data = response.data
        if user_data and user_data.get('password') == form.password.data:
            session['logged_in'] = True
            session['user_id'] = user_data.get('id')  # Simpan ID pengguna di sesi if perlu
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@main.route('/logout')
@login_required
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('main.login'))

@main.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/history')
@login_required
def history():
    return render_template('history.html')

@main.route('/run_test', methods=['GET', 'POST'])
@login_required
def run_test():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = file.filename
        
        if file:
            # Membaca data dari file CSV
            df = pd.read_csv(file, delimiter=',')
            df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d_%H:%M:%S.%f')
            
            # Algoritma KNN untuk Keputusan
            df['Master PV'] = df['Master PV'].apply(lambda x: float(str(x).replace(',', '.')))
            df['Condition'] = df['Master PV'].apply(lambda x: 'Aus' if x < 0 else 'Bagus')

            # Tambahkan kolom 'Titik' berdasarkan indeks data
            df['Titik'] = range(1, len(df) + 1)

            # Data pelatihan
            X = df[['Master PV']]
            y = df['Condition']

            # Membuat model KNN
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_scaled, y)

            # Prediksi untuk jarak tempuh hingga 36.525 KM
            additional_distance = 36525 - 24052
            df_pred = df.copy()
            df_pred['Titik'] = df_pred['Titik']
            df_pred['Master PV'] = df_pred['Master PV'].apply(lambda x: x * 0.95)  # Mengasumsikan penurunan PV sebesar 5%
            
            X_pred_scaled = scaler.transform(df_pred[['Master PV']])
            df_pred['Condition'] = knn.predict(X_pred_scaled)

            # Kondisi roda sebelum dan sesudah prediksi
            condition_before_prediction = df['Condition'].iloc[-1]
            condition_after_prediction = df_pred['Condition'].iloc[-1]

            # Grafik 1: Menampilkan data roda sejauh 24.052 KM
            plt.figure(figsize=(12, 6))
            ax1 = sns.lineplot(x=df['Titik'], y=df['Master PV'], marker='o')
            for i, (x, y) in enumerate(zip(df['Titik'], df['Master PV'])):
                ax1.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

            plt.title('Grafik Master PV (Jarak Tempuh 24.052 KM)')
            plt.xlabel('Titik')
            plt.ylabel('Master PV')
            ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Simpan grafik sebagai gambar PNG
            img1 = io.BytesIO()
            plt.savefig(img1, format='png')
            img1.seek(0)
            plot_url_1 = base64.b64encode(img1.getvalue()).decode()

            plt.clf()  # Bersihkan grafik sebelumnya untuk membuat yang baru

            # Grafik 2: Prediksi setelah jarak tempuh 36.525 KM
            plt.figure(figsize=(12, 6))
            ax2 = sns.lineplot(x=df_pred['Titik'], y=df_pred['Master PV'], marker='o')
            for i, (x, y) in enumerate(zip(df_pred['Titik'], df_pred['Master PV'])):
                ax2.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)

            plt.title('Grafik Prediksi Master PV (Jarak Tempuh 36.525 KM)')
            plt.xlabel('Titik')
            plt.ylabel('Master PV')
            ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Simpan grafik sebagai gambar PNG
            img2 = io.BytesIO()
            plt.savefig(img2, format='png')
            img2.seek(0)
            plot_url_2 = base64.b64encode(img2.getvalue()).decode()

            return render_template('run_test.html', plot_url_1=plot_url_1, plot_url_2=plot_url_2, 
                                   condition_before_prediction=condition_before_prediction, 
                                   condition_after_prediction=condition_after_prediction,
                                   filename=filename)

    return render_template('run_test.html')


