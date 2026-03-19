#include "quatOps.h"

Eigen::Matrix<double, 4, 1> quatType::rotToQuat(const Eigen::Matrix<double, 3, 3> &rot)
{
	Eigen::Matrix<double, 4, 1> q;
	double T = rot.trace();

	if ((rot(0, 0) >= T) && (rot(0, 0) >= rot(1, 1)) && (rot(0, 0) >= rot(2, 2)))
	{
		q(0) = sqrt((1 + (2 * rot(0, 0)) - T) / 4);
		q(1) = (1 / (4 * q(0))) * (rot(0, 1) + rot(1, 0));
		q(2) = (1 / (4 * q(0))) * (rot(0, 2) + rot(2, 0));
		q(3) = (1 / (4 * q(0))) * (rot(1, 2) - rot(2, 1));
	} 
	else if ((rot(1, 1) >= T) && (rot(1, 1) >= rot(0, 0)) && (rot(1, 1) >= rot(2, 2)))
	{
		q(1) = sqrt((1 + (2 * rot(1, 1)) - T) / 4);
		q(0) = (1 / (4 * q(1))) * (rot(0, 1) + rot(1, 0));
		q(2) = (1 / (4 * q(1))) * (rot(1, 2) + rot(2, 1));
		q(3) = (1 / (4 * q(1))) * (rot(2, 0) - rot(0, 2));
	}
	else if ((rot(2, 2) >= T) && (rot(2, 2) >= rot(0, 0)) && (rot(2, 2) >= rot(1, 1)))
	{
		q(2) = sqrt((1 + (2 * rot(2, 2)) - T) / 4);
		q(0) = (1 / (4 * q(2))) * (rot(0, 2) + rot(2, 0));
		q(1) = (1 / (4 * q(2))) * (rot(1, 2) + rot(2, 1));
		q(3) = (1 / (4 * q(2))) * (rot(0, 1) - rot(1, 0));
	} 
	else
	{
		q(3) = sqrt((1 + T) / 4);
		q(0) = (1 / (4 * q(3))) * (rot(1, 2) - rot(2, 1));
		q(1) = (1 / (4 * q(3))) * (rot(2, 0) - rot(0, 2));
		q(2) = (1 / (4 * q(3))) * (rot(0, 1) - rot(1, 0));
	}

	if (q(3) < 0)
	{
		q = -q;
	}

	q = q / (q.norm());
	return q;
}

// ============================================================================
// quatToRot: JPL四元数转旋转矩阵
// ============================================================================
// 功能：将JPL格式四元数 q = [x, y, z, w] 转换为 3×3 旋转矩阵
// 
// 数学推导：
// ─────────────────────────────────────────────────────────────────────────
// 1. 四元数基础
//    JPL四元数：q = [x, y, z, w] = [qx, qy, qz, qw]
//    其中：v = [x, y, z]^T 是虚部，w 是实部
//    单位四元数约束：||q|| = 1，即 x² + y² + z² + w² = 1
// 
// 2. Hamilton vs JPL 约定差异
//    Hamilton约定（右乘）：旋转公式 R = (2w² - 1)I + 2w[v]× + 2vv^T
//    JPL约定（左乘）：      旋转公式 R = (2w² - 1)I - 2w[v]× + 2vv^T
//                                                    ↑ 注意符号相反
// 
// 3. 公式推导
//    对于单位四元数：w² + x² + y² + z² = 1
//    因此：w² - (x² + y² + z²) = w² - (1 - w²) = 2w² - 1
// 
//    JPL旋转矩阵的三项分解：
//    ┌─────────────────────────────────────────────────────────────┐
//    │ R = (2w² - 1)I  -  2w[v]×  +  2vv^T                        │
//    │     ─────────     ────────     ──────                       │
//    │     缩放项        反对称项     秩一项                        │
//    └─────────────────────────────────────────────────────────────┘
// 
//    其中反对称矩阵（skew-symmetric）：
//              ⎡  0   -z    y ⎤
//    [v]× =    ⎢  z    0   -x ⎥
//              ⎣ -y    x    0 ⎦
// 
// 4. 展开形式（验证）
//    完整展开后的旋转矩阵：
//    ⎡ 1-2(y²+z²)    2(xy-wz)      2(xz+wy)   ⎤
//    ⎢ 2(xy+wz)      1-2(x²+z²)    2(yz-wx)   ⎥
//    ⎣ 2(xz-wy)      2(yz+wx)      1-2(x²+y²) ⎦
// 
// 5. 为什么JPL是负号？
//    JPL约定中，q 表示的是共轭Hamilton四元数的旋转效果
//    共轭操作：q* = [-x, -y, -z, w]
//    代入Hamilton公式：R = (2w² - 1)I + 2w[-v]× + 2(-v)(-v)^T
//                       = (2w² - 1)I - 2w[v]× + 2vv^T
//    这就是JPL公式！
// 
// 6. 物理意义
//    Hamilton: q ⊗ p 先应用q，再应用p（右到左读）
//    JPL:      q ⊗ p 先应用p，再应用q（右到左读，但乘法定义不同）
//    两者通过共轭关系联系：R_JPL(q) = R_Hamilton(q*)
// ============================================================================
Eigen::Matrix<double, 3, 3> quatType::quatToRot(const Eigen::Matrix<double, 4, 1> &q)
{
	// 步骤1：提取虚部 v = [x, y, z]，构造反对称矩阵 [v]×
	Eigen::Matrix<double, 3, 3> q_x = skewSymmetric(q.block<3, 1>(0, 0));
	
	// 步骤2：计算JPL旋转矩阵
	// R = (2w² - 1)I  -  2w[v]×  +  2vv^T
	//     ─────────     ────────     ──────
	//     term1         term2        term3
	Eigen::MatrixXd rot = (2 * std::pow(q(3, 0), 2) - 1) * Eigen::MatrixXd::Identity(3, 3)  // (2w² - 1)I
	                      - 2 * q(3, 0) * q_x                                                  // -2w[v]×
	                      + 2 * q.block<3, 1>(0, 0) * (q.block<3, 1>(0, 0).transpose());      // 2vv^T
	
	return rot;
}

Eigen::Matrix<double, 3, 3> quatType::skewSymmetric(const Eigen::Matrix<double, 3, 1> &w)
{
	Eigen::Matrix<double, 3, 3> w_x;
	w_x << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
	return w_x;
}

// ============================================================================
// quatMultiply: JPL四元数乘法（左乘约定）
// ============================================================================
// 功能: 计算两个JPL四元数的乘积 q ⊗ p
// 
// 参数:
//   q: 第一个四元数 [x, y, z, w] (JPL格式)
//   p: 第二个四元数 [x, y, z, w] (JPL格式)
// 
// 返回: 四元数乘积 q ⊗ p
// 
// 重要约定: quatMultiply(q, p) 使用左乘约定
//   - 表示先应用旋转 p，再应用旋转 q
//   - 即：R_result = R_q * R_p（矩阵左乘）
//   - 变换链：A → B → C，则使用 quatMultiply(q_B_to_C, q_A_to_B)
// 
// 实例验证（来自 initializer.cpp:731）:
//   ori_GtoIi = quatMultiply(ori_I0toIi, q_GtoI0)
//   其中 q_GtoI0 是 R_{I0←G}，ori_I0toIi 是 R_{Ii←I0}
//   变换链：G → I0 → Ii
//   矩阵形式：R_{Ii←G} = R_{Ii←I0} * R_{I0←G}
//   结果：ori_GtoIi 是 R_{Ii←G}（先应用 q_GtoI0，再应用 ori_I0toIi）
// ============================================================================
Eigen::Matrix<double, 4, 1> quatType::quatMultiply(const Eigen::Matrix<double, 4, 1> &q, const Eigen::Matrix<double, 4, 1> &p)
{
	Eigen::Matrix<double, 4, 1> q_t;
	Eigen::Matrix<double, 4, 4> Q_m;

	Q_m.block<3, 3>(0, 0) = q(3, 0) * Eigen::MatrixXd::Identity(3, 3) - skewSymmetric(q.block<3, 1>(0, 0));
	Q_m.block<3, 1>(0, 3) = q.block<3, 1>(0, 0);
	Q_m.block<1, 3>(3, 0) = -q.block<3, 1>(0, 0).transpose();
	Q_m(3, 3) = q(3, 0);
	q_t = Q_m * p;

	if (q_t(3, 0) < 0)
	{
		q_t *= -1;
	}

	return q_t / q_t.norm();
}

Eigen::Matrix<double, 3, 1> quatType::vee(const Eigen::Matrix<double, 3, 3> &w_x)
{
	Eigen::Matrix<double, 3, 1> w;
	w << w_x(2, 1), w_x(0, 2), w_x(1, 0);
	return w;
}

Eigen::Matrix<double, 3, 3> quatType::expSo3(const Eigen::Matrix<double, 3, 1> &w)
{
	Eigen::Matrix<double, 3, 3> w_x = skewSymmetric(w);
	double theta = w.norm();

	double A, B;
	if (theta < 1e-7) {
		A = 1;
		B = 0.5;
	}
	else
	{
		A = sin(theta) / theta;
		B = (1 - cos(theta)) / (theta * theta);
	}

	Eigen::Matrix<double, 3, 3> R;
	if (theta == 0) {
		R = Eigen::MatrixXd::Identity(3, 3);
	} else {
		R = Eigen::MatrixXd::Identity(3, 3) + A * w_x + B * w_x * w_x;
	}
	return R;
}

Eigen::Matrix<double, 3, 1> quatType::logSo3(const Eigen::Matrix<double, 3, 3> &R)
{
	double R_11 = R(0, 0), R_12 = R(0, 1), R_13 = R(0, 2);
	double R_21 = R(1, 0), R_22 = R(1, 1), R_23 = R(1, 2);
	double R_31 = R(2, 0), R_32 = R(2, 1), R_33 = R(2, 2);

	const double tr = R.trace();
	Eigen::Vector3d omega;

	if (tr + 1.0 < 1e-10)
	{
    	if (std::abs(R_33 + 1.0) > 1e-5)
			omega = (M_PI / sqrt(2.0 + 2.0 * R_33)) * Eigen::Vector3d(R_13, R_23, 1.0 + R_33);
		else if (std::abs(R_22 + 1.0) > 1e-5)
			omega = (M_PI / sqrt(2.0 + 2.0 * R_22)) * Eigen::Vector3d(R_12, 1.0 + R_22, R_32);
		else
			omega = (M_PI / sqrt(2.0 + 2.0 * R_11)) * Eigen::Vector3d(1.0 + R_11, R_21, R_31);
	}
	else
	{
		double magnitude;
		const double tr_3 = tr - 3.0;
		if (tr_3 < -1e-7)
		{
			double theta = acos((tr - 1.0) / 2.0);
			magnitude = theta / (2.0 * sin(theta));
		}
		else
		{
			magnitude = 0.5 - tr_3 / 12.0;
		}
		omega = magnitude * Eigen::Vector3d(R_32 - R_23, R_13 - R_31, R_21 - R_12);
	}
	return omega;
}

Eigen::Matrix4d quatType::expSe3(Eigen::Matrix<double, 6, 1> vec)
{
	Eigen::Vector3d w = vec.head(3);
	Eigen::Vector3d u = vec.tail(3);
	double theta = sqrt(w.dot(w));
	Eigen::Matrix3d wskew;
	wskew << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;

	double A, B, C;
	if (theta < 1e-7)
	{
		A = 1;
		B = 0.5;
		C = 1.0 / 6.0;
	}
	else
	{
		A = sin(theta) / theta;
		B = (1 - cos(theta)) / (theta * theta);
		C = (1 - A) / (theta * theta);
	}

	Eigen::Matrix3d I_33 = Eigen::Matrix3d::Identity();
	Eigen::Matrix3d V = I_33 + B * wskew + C * wskew * wskew;

	Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
	mat.block<3, 3>(0, 0) = I_33 + A * wskew + B * wskew * wskew;
	mat.block<3, 1>(0, 3) = V * u;
	mat(3, 3) = 1;
	return mat;
}

Eigen::Matrix<double, 6, 1> quatType::logSe3(Eigen::Matrix4d mat)
{
	Eigen::Vector3d w = logSo3(mat.block<3, 3>(0, 0));
	Eigen::Vector3d T = mat.block<3, 1>(0, 3);
	const double t = w.norm();
	if (t < 1e-10)
	{
		Eigen::Matrix<double, 6, 1> log;
		log << w, T;
		return log;
	}
	else
	{
		Eigen::Matrix3d W = skewSymmetric(w / t);
    	double Tan = tan(0.5 * t);
    	Eigen::Vector3d WT = W * T;
    	Eigen::Vector3d u = T - (0.5 * t) * WT + (1 - t / (2. * Tan)) * (W * WT);
    	Eigen::Matrix<double, 6, 1> log;
    	log << w, u;
    	return log;
	}
}

Eigen::Matrix4d quatType::hatSe3(const Eigen::Matrix<double, 6, 1> &vec)
{
	Eigen::Matrix4d mat = Eigen::Matrix4d::Zero();
	mat.block<3, 3>(0, 0) = skewSymmetric(vec.head(3));
	mat.block<3, 1>(0, 3) = vec.tail(3);
	return mat;
}

Eigen::Matrix4d quatType::invSe3(const Eigen::Matrix4d &T)
{
	Eigen::Matrix4d Tinv = Eigen::Matrix4d::Identity();
	Tinv.block<3, 3>(0, 0) = T.block<3, 3>(0, 0).transpose();
	Tinv.block<3, 1>(0, 3) = -Tinv.block<3, 3>(0, 0) * T.block<3, 1>(0, 3);
	return Tinv;
}

Eigen::Matrix<double, 4, 1> quatType::inv(Eigen::Matrix<double, 4, 1> q)
{
	Eigen::Matrix<double, 4, 1> q_inv;
	q_inv.block<3, 1>(0, 0) = -q.block<3, 1>(0, 0);
	q_inv(3, 0) = q(3, 0);
	return q_inv;
}

Eigen::Matrix<double, 4, 4> quatType::omega(Eigen::Matrix<double, 3, 1> w)
{
	Eigen::Matrix<double, 4, 4> mat;
	mat.block<3, 3>(0, 0) = -skewSymmetric(w);
	mat.block<1, 3>(3, 0) = -w.transpose();
	mat.block<3, 1>(0, 3) = w;
	mat(3, 3) = 0;
	return mat;
}

Eigen::Matrix<double, 4, 1> quatType::quatNorm(Eigen::Matrix<double, 4, 1> q_t)
{
	if (q_t(3, 0) < 0)
	{
		q_t *= -1;
	}

	return q_t / q_t.norm();
}

Eigen::Matrix<double, 3, 3> quatType::JleftSo3(const Eigen::Matrix<double, 3, 1> &w)
{
	double theta = w.norm();

	if (theta < 1e-6) {
		return Eigen::MatrixXd::Identity(3, 3);
	}
	else
	{
		Eigen::Matrix<double, 3, 1> a = w / theta;
		Eigen::Matrix<double, 3, 3> J = sin(theta) / theta * Eigen::MatrixXd::Identity(3, 3) + (1 - sin(theta) / theta) * a * a.transpose() +
										((1 - cos(theta)) / theta) * skewSymmetric(a);
		return J;
	}
}

Eigen::Matrix<double, 3, 3> quatType::JrighySo3(const Eigen::Matrix<double, 3, 1> &w)
{
	return JleftSo3(-w);
}

Eigen::Matrix<double, 3, 1> quatType::rotToRpy(const Eigen::Matrix<double, 3, 3> &rot)
{
	Eigen::Matrix<double, 3, 1> rpy;
	rpy(1, 0) = atan2(-rot(2, 0), sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0)));
	if (std::abs(cos(rpy(1, 0))) > 1.0e-12)
	{
		rpy(2, 0) = atan2(rot(1, 0) / cos(rpy(1, 0)), rot(0, 0) / cos(rpy(1, 0)));
		rpy(0, 0) = atan2(rot(2, 1) / cos(rpy(1, 0)), rot(2, 2) / cos(rpy(1, 0)));
	}
	else
	{
		rpy(2, 0) = 0;
		rpy(0, 0) = atan2(rot(0, 1), rot(1, 1));
	}
	return rpy;
}

Eigen::Matrix<double, 3, 3> quatType::rotX(double t)
{
	Eigen::Matrix<double, 3, 3> r;
	double ct = cos(t);
	double st = sin(t);
	r << 1.0, 0.0, 0.0, 0.0, ct, -st, 0.0, st, ct;
	return r;
}

Eigen::Matrix<double, 3, 3> quatType::rotY(double t)
{
	Eigen::Matrix<double, 3, 3> r;
	double ct = cos(t);
	double st = sin(t);
	r << ct, 0.0, st, 0.0, 1.0, 0.0, -st, 0.0, ct;
	return r;
}

Eigen::Matrix<double, 3, 3> quatType::rotZ(double t)
{
	Eigen::Matrix<double, 3, 3> r;
	double ct = cos(t);
	double st = sin(t);
	r << ct, -st, 0.0, st, ct, 0.0, 0.0, 0.0, 1.0;
	return r;
}