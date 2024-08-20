# app/routes.py
from flask import render_template, redirect, url_for, request, flash, current_app
from app.forms import RegistrationForm, LoginForm
from flask import Blueprint
from flask import session, redirect, url_for, flash, Blueprint
import csv
import io
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pandas as pd

main = Blueprint('main', __name__)

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
            flash('Login successful!', 'success')
            return redirect(url_for('main.dashboard'))
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', form=form)

@main.route('/logout')
def logout():
    # Hapus semua data dalam sesi
    session.clear()
    # Berikan pesan kepada pengguna bahwa mereka telah logout
    flash('You have been logged out.', 'info')
    # Arahkan kembali ke halaman login
    return redirect(url_for('main.login'))

@main.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/history')
def history():
    return render_template('history.html')

@main.route('/run_test', methods=['GET', 'POST'])
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
        
        if file:
            # Read the CSV file with the correct delimiter
            df = pd.read_csv(file, delimiter=',')  # Adjust delimiter if necessary

            # Convert 'Time' column to datetime format
            df['Time'] = pd.to_datetime(df['Time'], format='%Y-%m-%d_%H:%M:%S.%f')

            # Create a line plot
            plt.figure(figsize=(10,6))
            plt.plot(df['Time'], df['Master PV'], marker='o')
            plt.title('Master PV Over Time')
            plt.xlabel('Time')
            plt.ylabel('Master PV')
            plt.xticks(rotation=45)
            plt.tight_layout()

            # Save the plot as a PNG image
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()

            # Determine tire condition based on more negative or positive values
            jumlah_negatif = df['Master PV'].apply(lambda x: float(str(x).replace(',', '.'))).lt(0).sum()
            jumlah_positif = df['Master PV'].apply(lambda x: float(str(x).replace(',', '.'))).gt(0).sum()

            # Determine condition
            if jumlah_negatif > jumlah_positif:
                condition = 'Aus'
            else:
                condition = 'Baik'

            return render_template('run_test.html', plot_url=plot_url, condition=condition)
    
    return render_template('run_test.html')
