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
from sklearn.metrics import classification_report, confusion_matrix
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
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        # Get form inputs with default values
        current_distance_str = request.form.get('current_distance', '0')
        predicted_distance_str = request.form.get('predicted_distance', '0')
        assumption_decrease_str = request.form.get('assumption_decrease', '0')

        try:
            current_distance = float(current_distance_str)
            predicted_distance = float(predicted_distance_str)
            assumption_decrease = float(assumption_decrease_str) / 100
        except ValueError:
            flash('Invalid input for distance or assumption percentage.')
            return redirect(request.url)

        filename = file.filename
        
        if file:
            # Baca data dari file CSV
            df = pd.read_csv(file, delimiter=',')
            df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d_%H:%M:%S.%f')
            
            # Terapkan algoritma KNN untuk keputusan
            df['Master PV'] = df['Master PV'].apply(lambda x: float(str(x).replace(',', '.')))
            df['Condition Before'] = df['Master PV'].apply(lambda x: 'Aus' if x < 0 else 'Bagus')

            # Tambahkan kolom 'Titik' berdasarkan indeks data
            df['Titik'] = range(1, len(df) + 1)

            # Data pelatihan
            X = df[['Master PV']]
            y = df['Condition Before']

            # Buat model KNN
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_scaled, y)

            # Prediksi untuk jarak tempuh
            df_pred = df.copy()
            df_pred['Master PV'] = df['Master PV'] - (df['Master PV'].abs() * assumption_decrease)
            
            X_pred_scaled = scaler.transform(df_pred[['Master PV']])
            df_pred['Condition After'] = knn.predict(X_pred_scaled)

            # Menghitung jumlah "Bagus" dan "Aus" sebelum dan sesudah prediksi
            count_before = df['Condition Before'].value_counts()
            count_after = df_pred['Condition After'].value_counts()

            # Menghitung jumlah nilai negatif dan positif
            num_neg_before = (df['Master PV'] < 0).sum()
            num_pos_before = (df['Master PV'] >= 0).sum()
            num_neg_after = (df_pred['Master PV'] < 0).sum()
            num_pos_after = (df_pred['Master PV'] >= 0).sum()

            # Menentukan keadaan keseluruhan roda sebelum dan sesudah prediksi
            overall_condition_before = 'Bagus' if num_pos_before >= num_neg_before else 'Aus'
            overall_condition_after = 'Bagus' if num_pos_after >= num_neg_after else 'Aus'

            # Plot grafik
            plt.figure(figsize=(12, 6))
            sns.lineplot(x=df['Titik'], y=df['Master PV'], marker='o', label='Current')
            sns.lineplot(x=df_pred['Titik'], y=df_pred['Master PV'], marker='o', label='After Prediction')
            plt.title('Grafik Master PV Over Titik')
            plt.xlabel('Titik')
            plt.ylabel('Master PV')
            plt.legend()
            # Adding annotation for current distance
            plt.text(0.05, 0.95, f'Jarak Sekarang: {current_distance} km', transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

            # Adding annotation for predicted distance
            plt.text(0.05, 0.90, f'Jarak Prediksi: {predicted_distance} km', transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

            # Adding annotation for assumption decrease
            plt.text(0.05, 0.85, f'Asumsi Penurunan: {assumption_decrease * 100:.0f}%', transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))
            plt.tight_layout()

            # Menambahkan anotasi nilai pada grafik
            for i, (titik, pv_value) in enumerate(zip(df['Titik'], df['Master PV'])):
                plt.text(titik, pv_value, f'{pv_value:.2f}', fontsize=9, ha='right', va='bottom', color='blue')

            for i, (titik, pv_value) in enumerate(zip(df_pred['Titik'], df_pred['Master PV'])):
                plt.text(titik, pv_value, f'{pv_value:.2f}', fontsize=9, ha='right', va='top', color='red')

            # Mengatur jarak untuk menghindari tumpang tindih
            plt.gca().margins(x=0.05, y=0.15)
            plt.tight_layout()

            # Save plot as PNG image
            img1 = io.BytesIO()
            plt.savefig(img1, format='png')
            img1.seek(0)
            plot_url_1 = base64.b64encode(img1.getvalue()).decode()

            plt.clf()

            return render_template(
                'run_test.html',
                plot_url_1=plot_url_1,
                filename=filename,
                condition_before_prediction=overall_condition_before,
                condition_after_prediction=overall_condition_after,
                predicted_distance = predicted_distance
            )

    return render_template('run_test.html')

