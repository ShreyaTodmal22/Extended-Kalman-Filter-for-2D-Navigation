% initializing state vector
x0=-9.764;% simultaneous solving for two equations
y0=-9.764;
vx0=0; % asuming velocity to be zero at the start
vy0=0;

Lb1 = [9.72; 9.72; 22.08; 22.08];
Lb2 = [8.77;8.77;20.74;20.74];
Lb3 = [7.97;7.97;19.42;19.42];
Lb4 = [7.34;7.34;18.10;18.10];
Lb5 = [6.93;6.93;16.81;16.81];
Lb6 = [6.79;6.79;15.53;15.53];
Lb7 = [6.93;6.93;14.28;14.28];
Lb8 = [7.34;7.34;13.06;13.06];
Lb9 = [7.97;7.97;11.88;11.88];
Lb10 = [8.77;8.77;10.76;10.76];
Lb11 = [9.72;9.72;9.72;9.72];
Lb12 = [10.76;10.76;8.77;8.77];
Lb13 = [11.88;11.88;7.97;7.97];
Lb14 = [13.06;13.06;7.43;7.34];
Lb15 = [14.28;14.28;6.93;6.93];
Lb16 = [15.53;15.53;6.79;6.79];
Lb17 = [16.81;16.81;6.93;6.93];
Lb18 = [18.10;18.10;7.34;7.43];
Lb19 = [19.42;19.42;7.97;7.97];
Lb20 = [20.74;20.74;8.77;8.77];
Lb21 = [22.08;22.08;9.72;9.72];

% Define observation matrices in a cell array
Lb_cell = {Lb1, Lb2, Lb3, Lb4, Lb5, Lb6, Lb7, Lb8, Lb9, Lb10, Lb11, Lb12, Lb13, Lb14, Lb15, Lb16, Lb17, Lb18, Lb19, Lb20, Lb21};

% Initializing covariance matrices
Pk_1 = diag([1,1,1,1]); % Covariance of the process noise
Qk_1 = diag([0.001,0.001,0.001,0.001]); % Covariance of the state estimate
Rk_1 = diag([1, 1, 1, 1]); % Covariance of the observation noise

% Initialize lists to store final coordinates
x_list = [];
y_list = [];
trace_P=[];

for i=1:21

    % define state vector
    Xk_1=[x0;y0;vx0;vy0];

    % define Measurement model
    Zk=Lb_cell{i};

    % define time step
    dt=1;

    % define Fk matrix
    Fk=[1 0 dt 0;
        0 1 0 dt;
        0 0 1 0;
        0 0 0 1];

    % Kalman filter
    % prediction
    Xk_predicted=Fk*Xk_1; % Prediction of state
    Pk_predicted=(Fk*Pk_1*Fk')+Qk_1; % Prediction of covariance

    % State transition model
    x0=x0+vx0*dt;
    y0=y0+vy0*dt;
    vx0=vx0; % assuming velocity changes slowly
    vy0=vy0;
    
    % defining measurement equations 
    syms x y vx vy
    d10 = sqrt((-10-x)^2+(0-y)^2);
    d20 = sqrt((0-x)^2+(-10-y)^2);
    d30 = sqrt((10-x)^2+(0-y)^2);
    d40 = sqrt((0-x)^2+(10-y)^2);

    % Design matrix H
    Hk = jacobian([d10,d20,d30,d40], [x, y, vx, vy]);
    Hk = double(subs(Hk, [x, y, vx, vy], [Xk_predicted(1,1),Xk_predicted(2,1),Xk_predicted(3,1),Xk_predicted(4,1)]));
    %Hk = double(subs(Hk, [x, y, vx, vy], [x0,y0,vx0,vy0]));

    % defininf f(X0)
    f_x0=[d10;d20;d30;d40];
    f_x0 = double(subs(f_x0, [x, y], [x0,y0]));

    % Measurement model
    Zk_=Hk*Xk_predicted+f_x0-Hk*Xk_1;

    %kalman gain
    Kk=Pk_predicted*Hk'*inv(Hk*Pk_predicted*Hk'+Rk_1);

    % updating
    Xk=Xk_predicted+Kk*(Zk-Zk_); % state vector
    Pk=(eye(4)-Kk*Hk)*Pk_predicted; % error covariance

    % updating variable
    Pk_1=Pk;
    x0=Xk(1,1);
    y0=Xk(2,1);
    vx0=Xk(3,1);
    vy0=Xk(4,1);

    disp(['Coordinates of car at position', num2str(i)]);
    disp(x0)
    disp(y0)
    % Store the final coordinates in lists
    x_list = [x_list; x0];
    y_list = [y_list; y0];

    % storing variables for plotting
    trace_P(i)=trace(Pk);
end

% Define control points 
x_control = [-10, 0, 10, 0];
y_control = [0, -10, 0, 10];

% Plot a graph between x and y
figure;
hold on;
plot(x_list, y_list, 'o', 'DisplayName', 'Final Coordinates');
plot(x_control, y_control, 's', 'DisplayName', 'Control Points');
plot(x_list, y_list, '-', 'Color', 'b', 'DisplayName', 'Trajectory of car')
hold off;
xlabel('x');
ylabel('y');
title('Trajectory of the Car');
legend;
xlim([-10, 10]); % set x-axis limits to -10 and 10
ylim([-10, 10]); % set y-axis limits to -10 and 10

% Plotting Trace of Estimation Error Covariance 
figure;
dt = 0:20;
plot(dt, trace_P, 'm', 'LineWidth', 2); % Trace of estimation error covariance
xlabel('Time (1 sec interval)');
ylabel('Trace of P');
title('Trace of Estimation Error Covariance (P) vs Time');
grid on;
