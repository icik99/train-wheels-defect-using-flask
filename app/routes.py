from flask import render_template, redirect, url_for, request, flash, current_app, session
from app.forms import RegistrationForm, LoginForm
from flask import Blueprint
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
from werkzeug.security import generate_password_hash
import numpy as np
from scipy.interpolate import make_interp_spline

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
        diameter_roda_str = request.form.get('diameter_roda', '0')

        try:
            current_distance = float(current_distance_str)
            diameter_roda = float(diameter_roda_str)

            # Calculate assumption_decrease using the given formula
            penurunanDiameter = 20 / current_distance
            assumption_decrease = (penurunanDiameter / diameter_roda) * 100
            penurunanAsumsiDalamDesimal = assumption_decrease / 100
        except ValueError:
            flash('Invalid input for distance or wheel diameter.')
            return redirect(request.url)


        filename = file.filename
        
        if file:
            # Read the Excel file
            excel_file = pd.ExcelFile(file)

            # Initialize the new DataFrame
            new_df = pd.DataFrame(columns=['Titik', 'Time', 'Master PV'])

            for idx, sheet_name in enumerate(excel_file.sheet_names):
                sheet_df = excel_file.parse(sheet_name)

                try:
                    # Periksa apakah kolom yang diharapkan ada
                    if 'Time' not in sheet_df.columns or 'Master PV' not in sheet_df.columns:
                        print(f"Skipping sheet {sheet_name}: Missing required columns")
                        continue

                    # Ekstraksi data dari baris pertama kolom 'Time' dan 'Master PV'
                    time_value = sheet_df['Time'].iloc[0]
                    master_pv_value = sheet_df['Master PV'].iloc[0]

                    # Validasi dan konversi data
                    time_converted = pd.to_datetime(time_value, format='%Y-%m-%d_%H:%M:%S.%f', errors='coerce')


                    # Coba konversi nilai 'Master PV' ke float
                    try:
                        master_pv_converted = float(str(master_pv_value).replace(',', '.'))
                    except ValueError:
                        master_pv_converted = None
                    


                    # Jika konversi berhasil, tambahkan ke DataFrame baru
                    if pd.notnull(time_converted) and master_pv_converted is not None:
                        new_df = new_df._append({
                            'Titik': idx + 1,
                            'Time': time_converted,
                            'Master PV': master_pv_converted
                        }, ignore_index=True)
                    else:
                        print(f"Skipping sheet {sheet_name}: Invalid data format")

                except Exception as e:
                    print(f"Error processing sheet {sheet_name}: {e}")
                    # Lewati sheet ini dan lanjutkan ke sheet berikutnya
                    continue

            # Continue with your existing logic using the new_df instead of df
            df = new_df
            df['Condition Before'] = df['Master PV'].apply(lambda x: 'Aus' if x < 0 else 'Bagus')

            predicted_distance = current_distance
            max_iterations = 9999999  # Maximum number of iterations to prevent infinite loop
            iteration = 0
            df_pred = df.copy()
            while iteration < max_iterations:
                # Calculate the assumption decrease percentage based on the current distance
                # Apply assumption decrease
                df_pred['Master PV'] = df_pred['Master PV'] - (df['Master PV'].abs() * (penurunanAsumsiDalamDesimal))
                
                # Hitung jumlah data positif dan negatif
                nilaiPositifSetelahPrediksi = (df_pred['Master PV'] >= 0).sum()
                nilaiNegatifSetelahPrediksi = (df_pred['Master PV'] < 0).sum()

                # Cek apakah jumlah data negatif lebih banyak daripada data positif
                if nilaiNegatifSetelahPrediksi > nilaiPositifSetelahPrediksi:
                    break

                # Increment the distance to simulate further wear
                predicted_distance += 1  # Increment by 1 km or adjust as needed

                iteration += 1
                print(f'Iterasi: {iteration}')
                print(f'Prediksi jarak: {predicted_distance}')
                print(df_pred['Master PV'])

            if iteration >= max_iterations:
                print('The prediction did not converge within the maximum number of iterations.')
                print('DATA AWAL:')
                print(df['Master PV'])
                print('DATA SETELAH PREDIKSI:')
                print(df_pred['Master PV'])
                print('Asumsi penurunan: ', assumption_decrease)
                
            # Remaining logic remains the same...
            X = df[['Master PV']]
            y = df['Condition Before']

            # Scaling and KNN logic
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(X_scaled, y)
            X_pred_scaled = scaler.transform(df_pred[['Master PV']])
            df_pred['Condition After'] = knn.predict(X_pred_scaled)

            # Menghitung jumlah nilai negatif dan positif
            num_neg_before = (df['Master PV'] < 0).sum()
            num_pos_before = (df['Master PV'] >= 0).sum()
            num_neg_after = (df_pred['Master PV'] < 0).sum()
            num_pos_after = (df_pred['Master PV'] >= 0).sum()

            # Menentukan keadaan keseluruhan roda sebelum dan sesudah prediksi
            overall_condition_before = 'Bagus' if num_pos_before >= num_neg_before else 'Aus'
            overall_condition_after = 'Bagus' if num_pos_after >= num_neg_after else 'Aus'

            # Negative points identification and table creation (before and after)
            # Identify negative points and apply styling (current)
            negative_points_before = df[df['Master PV'] < 0][['Titik', 'Master PV']]
            negative_points_before_styled = negative_points_before.style \
                .set_table_attributes('class="table table-bordered table-striped table-hover table-sm"') \
                .format() \
                .set_properties(**{
                    'text-align': 'center',
                    'background-color': '#f9f9f9',
                    'color': '#333',
                    'border': '1px solid #ddd',
                    'padding': '14px'  # Adding padding for readability
                }) \
                .applymap(lambda x: 'background-color: #ffcccc' if isinstance(x, (int, float)) and x < 0 else '', subset=['Master PV']) \
                

            negative_points_before_html = negative_points_before_styled.hide(axis='index').to_html()

            # Identify negative points and apply styling (after prediction)
            negative_points_after = df_pred[df_pred['Master PV'] < 0][['Titik', 'Master PV']]
            negative_points_after_styled = negative_points_after.style \
                .set_table_attributes('class="table table-bordered table-striped table-hover table-sm"') \
                .format() \
                .set_properties(**{
                    'text-align': 'center',
                    'background-color': '#f9f9f9',
                    'color': '#333',
                    'border': '1px solid #ddd',
                    'padding': '14px'  # Adding padding for readability
                }) \
                .applymap(lambda x: 'background-color: #ffcccc' if isinstance(x, (int, float)) and x < 0 else '', subset=['Master PV']) \

            negative_points_after_html = negative_points_after_styled.hide(axis='index').to_html()

            x_current = df['Titik']
            y_current = df['Master PV']

            x_pred = df_pred['Titik']
            y_pred = df_pred['Master PV']

            # Interpolasi dengan spline untuk membuat kurva lebih halus
            xnew_current = np.linspace(x_current.min(), x_current.max(), 300)  # 300 titik untuk interpolasi
            y_smooth_current = make_interp_spline(x_current, y_current, k=3)(xnew_current)

            xnew_pred = np.linspace(x_pred.min(), x_pred.max(), 300)  # 300 titik untuk interpolasi
            y_smooth_pred = make_interp_spline(x_pred, y_pred, k=3)(xnew_pred)

            plt.figure(figsize=(12, 6))

            # Plot interpolated lines
            plt.plot(xnew_current, y_smooth_current, label='Current', color='blue', linewidth=1)
            plt.plot(xnew_pred, y_smooth_pred, label='After Prediction', color='orange', linewidth=1)

            # Highlight negative points before prediction (current)
            plt.scatter(df['Titik'], df['Master PV'], color='blue', marker='o', edgecolor='black', s=50)
            for titik, pv_value in zip(df['Titik'], df['Master PV']):
                if pv_value < 0:
                    plt.plot(titik, pv_value, 'ro', markersize=9)  # Red color for negative points before prediction

            # Highlight negative points after prediction
            plt.scatter(df_pred['Titik'], df_pred['Master PV'], color='orange', marker='o', edgecolor='orange', s=50)
            for titik, pv_value in zip(df_pred['Titik'], df_pred['Master PV']):
                if pv_value < 0:
                    plt.plot(titik, pv_value, 'ro', markersize=9)  # Red 'x' marker for negative points after prediction

            # Add custom legend for wear warnings (ke-ausan)
            plt.plot([], [], 'ro', label='Warning: Aus')  # Legend for red circles (current negative points)
            plt.plot([], [], 'ro', label='Warning After Prediction: Aus')  # Legend for red 'x' (predicted negative points)

            plt.title('Grafik Master PV Over Titik')
            plt.xlabel('Titik')
            plt.ylabel('Master PV')
            plt.legend()

            # Adding grid for better readability
            plt.grid(True, linestyle='--', alpha=0.7)

            # Set the ticks for the x-axis and y-axis
            plt.xticks(np.arange(min(df['Titik'].min(), df_pred['Titik'].min()), max(df['Titik'].max(), df_pred['Titik'].max()) + 1, 1))

            # Adding annotation for current distance
            plt.text(0.05, 0.95, f'Jarak Sekarang: {current_distance} km', transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

            # Adding annotation for predicted distance
            plt.text(0.05, 0.90, f'Jarak Prediksi Akan Terjadi Aus: {predicted_distance} km', transform=plt.gca().transAxes,
                    fontsize=12, verticalalignment='top', color='black', bbox=dict(facecolor='white', alpha=0.5))

            # Adding annotation values on the plot
            for i, (titik, pv_value) in enumerate(zip(df['Titik'], df['Master PV'])):
                plt.text(titik, pv_value, f'{pv_value:.2f}', fontsize=9, ha='right', va='bottom', color='blue')

            for i, (titik, pv_value) in enumerate(zip(df_pred['Titik'], df_pred['Master PV'])):
                plt.text(titik, pv_value, f'{pv_value:.2f}', fontsize=9, ha='right', va='top', color='orange')

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
                predicted_distance=predicted_distance,
                current_distance = current_distance,
                negative_points_before_table=negative_points_before_html,
                negative_points_after_table=negative_points_after_html,
                assumption_decrease = assumption_decrease
            )

    return render_template('run_test.html')



