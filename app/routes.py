from flask import render_template, redirect, url_for, request, flash, current_app, session
from app.forms import RegistrationForm, LoginForm
from flask import Blueprint
import io
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import base64
from io import BytesIO
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import classification_report, confusion_matrix
from werkzeug.security import generate_password_hash
import numpy as np
from scipy.interpolate import make_interp_spline
import math

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

# Register
@main.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            supabase = current_app.config['SUPABASE_CLIENT']
            user = supabase.table('users').insert({
                "name": form.name.data,
                "email": form.email.data,
                "phone": form.phone.data,
                "password": generate_password_hash(form.password.data)  # Hashing password
            }).execute()
            
            flash('Account created successfully!', 'success')
            return redirect(url_for('main.login'))
        except Exception as e:
            # Menangani kesalahan saat penyimpanan data
            flash(f'An error occurred: {str(e)}', 'danger')
    else:
        # Menampilkan kesalahan validasi
        flash('Please correct the errors in the form.', 'danger')
        print(form.errors)  # Log kesalahan validasi di console

    return render_template('register.html', form=form)


# Fungsi Untuk Login
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
@login_required
def about():
    return render_template('about.html')

@main.route('/history')
@login_required
def history():
    return render_template('history.html')



@main.route('/run_test', methods=['GET', 'POST'])
@login_required
def run_test():

    # Fungsi untuk Visualisasi Grafik
    def create_plot(x_current, y_current, x_pred, y_pred, sisi, current_distance, predicted_distance):
        plt.figure(figsize=(12, 6))

        # Interpolasi dengan spline untuk membuat kurva lebih halus
        xnew_current = np.linspace(x_current.min(), x_current.max(), 300)
        y_smooth_current = make_interp_spline(x_current, y_current, k=3)(xnew_current)

        xnew_pred = np.linspace(x_pred.min(), x_pred.max(), 300)
        y_smooth_pred = make_interp_spline(x_pred, y_pred, k=3)(xnew_pred)

        # Plot interpolated lines
        plt.plot(xnew_current, y_smooth_current, label='Current', color='blue', linewidth=1)
        plt.plot(xnew_pred, y_smooth_pred, label='After Prediction', color='orange', linewidth=1)

        # Highlight points before prediction (current)
        plt.scatter(x_current, y_current, color='blue', marker='o', edgecolor='black', s=50)
        for x, y in zip(x_current, y_current):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', color='blue')

        # Highlight points after prediction
        plt.scatter(x_pred, y_pred, color='orange', marker='o', edgecolor='orange', s=50)
        for x, y in zip(x_pred, y_pred):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,5), ha='center', color='orange')

        # Add early warning markers for negative points
        plt.scatter(x_current[y_current < 0], y_current[y_current < 0], color='red', marker='o', s=100, label='Current Negative')
        plt.scatter(x_pred[y_pred < 0], y_pred[y_pred < 0], color='red', marker='o', s=100, label='Predicted Negative')

        plt.title(f'Grafik {sisi} Over Titik')
        plt.xlabel('Titik')
        plt.ylabel(f'{sisi}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(np.arange(min(x_current.min(), x_pred.min()), max(x_current.max(), x_pred.max()) + 1, 1))

        # Adding annotation for current distance
        plt.text(0.05, 0.95, f'Jarak Sekarang: {current_distance} km', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

        # Adding annotation for predicted distance
        plt.text(0.05, 0.90, f'Jarak Prediksi Akan Terjadi Aus: {predicted_distance} km', transform=plt.gca().transAxes,
                fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

        plt.gca().margins(x=0.05, y=0.15)
        plt.tight_layout()

        # Save plot as PNG image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plt.close()

        return base64.b64encode(img.getvalue()).decode()


    conditions_before = {}
    conditions_after = {}
    predicted_distances = {}
    graphs_urls = {}
    negative_values = {}
    filename = None
    current_distance = None
    assumption_decrease = None
    penurunanAsumsiDalamDesimal = None

    if request.method == 'POST':
        # Untuk Nangkap Inputan dari Halaman Run Test (Csv, Jarak Tempuh Roda Sekarang, dan Diameter Roda)
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        current_distance_str = request.form.get('current_distance', '0')
        diameter_roda_str = request.form.get('diameter_roda', '0')
        
        try:
            current_distance = float(current_distance_str)
            diameter_roda = float(diameter_roda_str)

            # Rumus Asumsi Penurunan Per Kilometer
            penurunanDiameter = 20 / current_distance
            assumption_decrease = (penurunanDiameter / diameter_roda) * 100
            penurunanAsumsiDalamDesimal = assumption_decrease / 100

            # Menentukan ambang batas untuk membulatkan nilai
            batas = 0.01
            if penurunanAsumsiDalamDesimal < batas:
                penurunanAsumsiDalamDesimal = 0.01

        except ValueError:
            flash('Invalid input for distance or wheel diameter.')
            return redirect(request.url)

        filename = file.filename

        try:
            if file:
                try:
                    df = pd.read_csv(file)  # Membaca CSV
                except FileNotFoundError:
                    return "File tidak ditemukan. Pastikan file diunggah dengan benar."
                except pd.errors.EmptyDataError:
                    return "File CSV kosong. Mohon unggah file yang valid."
                except pd.errors.ParserError:
                    return "Terjadi kesalahan saat membaca file. Pastikan format CSV sudah benar."
                
                try:
                    # Membuat label kondisi sebelum (misalnya, menggunakan Sisi 1 sebagai contoh)
                    df['Condition_Before'] = df['Sisi 1'].apply(lambda x: 'Bagus' if x >= 0 else 'Aus')
                except KeyError as e:
                    return f"Kolom yang diminta tidak ditemukan: {e}. Pastikan CSV memiliki kolom yang benar."
                
                try:
                    # Membagi dataset menjadi fitur (X) dan label (y)
                    X = df[['Titik', 'Sisi 1', 'Sisi 2', 'Sisi 3', 'Sisi 4']]
                    y = df['Condition_Before']

                    # Membagi data menjadi data training dan data uji, Data Training sebanyak 80%, dan data uji sebanyak 20%
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                except KeyError as e:
                    return f"Terjadi kesalahan saat membagi dataset: {e}. Pastikan kolom yang dibutuhkan ada di dataset."
                except ValueError as e:
                    return f"Terjadi kesalahan saat membagi dataset: {e}. Pastikan data cukup untuk dibagi menjadi data training dan testing."

                try:
                    # Inisialisasi dan melatih model KNN
                    knn = KNeighborsClassifier(n_neighbors=3)
                    knn.fit(X_train, y_train)
                except Exception as e:
                    return f"Terjadi kesalahan saat melatih model KNN: {str(e)}"

                threshold = 1e-10  # Threshold untuk menganggap nilai sangat kecil sebagai nol

                # Fungsi untuk memeriksa, apakah titik tergolong aus atau bagus
                def determine_condition(series):
                    nilai_positif = (series >= 0).sum()
                    nilai_negatif = (series < 0).sum()
                    return 'Bagus' if nilai_positif >= nilai_negatif else 'Aus'

                # Perhitungan Prediksi Jarak Tempuh
                for sisi in ['Sisi 1', 'Sisi 2', 'Sisi 3', 'Sisi 4']:
                    df_pred = df.copy()
                    predicted_distance = current_distance # ini inisasi awal jarak prediksi
                    max_iterations = 8000
                    iteration = 0
                    print('menghitung', sisi + '....')

                    try:

                        while iteration < max_iterations: #Perulangan akan terus jalan sampai iterasi maksimal

                            # Perhitungan untuk mengurangi nilai pv yang ada, dengan asumsi penurunan per kilometer.
                            df_pred[sisi] = (df_pred[sisi] - (df[sisi].abs() * penurunanAsumsiDalamDesimal))
                            
                            # Mengganti nilai yang sangat kecil menjadi nol
                            df_pred[sisi] = np.where(np.abs(df_pred[sisi]) < threshold, 0, df_pred[sisi])
                            

                            nilaiPositifSetelahPrediksi = (df_pred[sisi] >= 0).sum()
                            nilaiNegatifSetelahPrediksi = (df_pred[sisi] < 0).sum()

                            if nilaiNegatifSetelahPrediksi > nilaiPositifSetelahPrediksi:
                                print(sisi, 'berhasil di hitung')
                                break

                            if (df_pred[sisi] <= -10).any():
                                print(f"Ada nilai {sisi} yang mencapai -10 atau kurang. Menghentikan perulangan.")
                                break

                            predicted_distance += 1
                            iteration += 1
                            print('Menghitung ' + sisi + ' | Prediksi Jarak = ' , predicted_distance, ' Kilometer')
                    except Exception as e:
                        return f"Terjadi kesalahan saat menghitung prediksi pada {sisi}: {str(e)}"

                    try:
                        # Membulatkan nilai asli dan nilai prediksi dengan 2 angka di belakang koma
                        df[sisi] = df[sisi].round(2)
                        df_pred[sisi] = df_pred[sisi].round(2)

                        # Prediksi kondisi sebelum menggunakan KNN
                        conditions_before[sisi] = 'Bagus' if (df[sisi] >= 0).sum() >= (df[sisi] < 0).sum() else 'Aus'
                        
                        # Menentukan kondisi sebelum dan setelah prediksi
                        conditions_before[sisi] = determine_condition(df[sisi])
                        conditions_after[sisi] = determine_condition(df_pred[sisi])

                        predicted_distances[sisi] = predicted_distance
                        # Ambil nilai-nilai negatif
                        negative_values[sisi] = df_pred[df_pred[sisi] < 0].copy()
                        negative_values[sisi]['Titik'] = negative_values[sisi]['Titik'].astype(int)

                        x_current = df['Titik']
                        y_current = df[sisi]
                        x_pred = df_pred['Titik']
                        y_pred = df_pred[sisi]
                        
                        graphs_urls[sisi] = create_plot(x_current, y_current, x_pred, y_pred, sisi, current_distance, predicted_distance)
                    except KeyError as e:
                        return f"Kolom yang diminta tidak ditemukan: {e}. Pastikan semua kolom yang dibutuhkan ada di dataset."
                    except Exception as e:
                        return f"Terjadi kesalahan dalam proses perhitungan atau plotting: {str(e)}"
        except Exception as e:
            return f"Terjadi kesalahan yang tidak terduga: {str(e)}"

    # ini untuk return / mengembalikan nilai hasil perhitungan dan prediksi ke website
    return render_template(
        'run_test.html',
        graphs_urls=graphs_urls,
        filename=filename,
        conditions_before=conditions_before,
        conditions_after=conditions_after,
        predicted_distances=predicted_distances,
        current_distance=current_distance,
        assumption_decrease=assumption_decrease,
        negative_values = negative_values
    )

    return render_template('run_test.html')


