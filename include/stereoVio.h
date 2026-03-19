#pragma once

// c++ include
#include <iostream>
#include <iomanip>
#include <cmath>
#include <math.h>
#include <thread>
#include <fstream>
#include <vector>
#include <queue>
#include <unordered_set>
#include <tsl/robin_map.h>

// lib include
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <sensor_msgs/Imu.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp> 
#include <Eigen/Core>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

// function include
#include "utility.h"
#include "parameters.h"
#include "sensorData.h"
#include "state.h"
#include "stateHelper.h"
#include "msckf.h"
#include "feature.h"
#include "featureTracker.h"
#include "mapPoint.h"
#include "mapManagement.h"
#include "initializer.h"
#include "frame.h"

struct Measurements
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<imuData> imu_measurements;
    cameraData image_measurements;
};

struct GTPose {
    double timestamp;
    Eigen::Quaterniond q;
    Eigen::Vector3d t;
};

// ==================== 多帧射线投射查询相关结构体 ====================

/**
 * @brief 3D射线结构体
 * 
 * 用于多帧射线投射查询中表示从相机中心发出的射线。
 * 包含射线的几何信息和来源帧索引。
 */
struct Ray3D {
    Eigen::Vector3d origin;         // 射线起点（相机中心在世界坐标系中的位置）
    Eigen::Vector3d direction;      // 射线方向（单位向量，世界坐标系）
    size_t frame_index;             // 射线来源的帧索引（用于调试和追踪）
    
    /**
     * @brief 默认构造函数
     * 初始化frame_index为0，origin和direction使用Eigen默认构造（零向量）
     */
    Ray3D() : frame_index(0) {}
    
    /**
     * @brief 带参数的构造函数
     * @param orig 射线起点
     * @param dir 射线方向
     * @param idx 帧索引
     */
    Ray3D(const Eigen::Vector3d& orig, const Eigen::Vector3d& dir, size_t idx = 0)
        : origin(orig), direction(dir), frame_index(idx) {}
};

/**
 * @brief 体素搜索区域结构体
 * 
 * 定义多帧射线投射查询中的约束搜索区域，用于优化DDA遍历效率。
 * 通过分析多条射线的交汇几何来确定最有可能包含真实交点的空间区域。
 */
struct VoxelSearchRegion {
    Eigen::Vector3d center;         // 搜索区域的中心点坐标（世界坐标系），通常是多条射线交汇点的平均位置
    double radius;                  // 搜索区域的半径（米），定义了以center为中心的球形搜索范围
    double min_depth;               // 最小搜索深度（米），射线投射的起始距离，避免搜索过近的体素
    double max_depth;               // 最大搜索深度（米），射线投射的终止距离，限制搜索范围以提高效率
    double optimal_resolution;      // 针对此搜索区域推荐的最佳体素分辨率（米），基于深度和投影尺寸计算得出
    bool is_valid;                  // 搜索区域是否有效的标志位，false表示射线束无法形成有效的交汇区域
    
    /**
     * @brief 默认构造函数
     * 
     * 初始化所有参数为安全的默认值：
     * - radius = 0: 无搜索范围
     * - min_depth = 0: 从相机位置开始
     * - max_depth = 0: 无搜索深度
     * - optimal_resolution = 0.1f: 使用最高精度分辨率
     * - is_valid = false: 标记为无效状态，需要通过计算来验证
     */
    VoxelSearchRegion() : radius(0), min_depth(0), max_depth(0), optimal_resolution(0.1f), is_valid(false) {}
    
    /**
     * @brief 带参数的构造函数
     * @param c 搜索中心
     * @param r 搜索半径
     * @param min_d 最小深度
     * @param max_d 最大深度
     * @param res 最佳分辨率
     */
    VoxelSearchRegion(const Eigen::Vector3d& c, double r, double min_d, double max_d, double res)
        : center(c), radius(r), min_depth(min_d), max_depth(max_d), optimal_resolution(res), is_valid(true) {}
};

/**
 * 多分辨率体素地图键值结构体（整数键版本 - 避免浮点精度问题）
 * 
 * 该结构体定义了多分辨率体素地图中每个体素的唯一标识符。
 * 使用整数索引代替浮点分辨率，完全避免精度问题。
 * 
 * 分辨率映射（厘米精度）：0.01m→1, 0.015m→1.5→2, 0.02m→2, 0.05m→5, ..., 1.0m→100
 */
struct MultiResVoxelKey {
    int x, y, z;                    // 体素的三维网格索引坐标（相对于原点的网格位置）
    int resolution_index;           // 分辨率索引（厘米级，1-100，对应 0.01m-1.0m）
    
    /**
     * @brief 构造函数（从浮点分辨率转换为整数索引）
     * @param x_ X轴网格索引
     * @param y_ Y轴网格索引  
     * @param z_ Z轴网格索引
     * @param res_ 体素分辨率（米），范围 0.01-1.0
     */
    MultiResVoxelKey(int x_, int y_, int z_, double res_) 
        : x(x_), y(y_), z(z_) {
        // 将分辨率（米）映射到厘米索引（精度到1cm）
        // 0.01 -> 1, 0.015 -> 1.5 -> 2, 0.02 -> 2, 0.05 -> 5, ..., 1.0 -> 100
        resolution_index = static_cast<int>(std::round(res_ * 100.0));
        
        // 边界检查，确保在有效范围内 (支持 0.01m-1.0m)
        if (resolution_index < 1) resolution_index = 1;
        if (resolution_index > 100) resolution_index = 100;  // 最大1.0m
    }
    
    /**
     * @brief 获取实际分辨率值（米）
     * @return 分辨率浮点值
     */
    double getResolution() const {
        return static_cast<double>(resolution_index) / 100.0;
    }
    
    /**
     * @brief 小于运算符，用于std::map的有序存储
     * 
     * 排序优先级：resolution_index > x > y > z
     * 
     * @param other 待比较的另一个体素键
     * @return true 如果当前键小于other
     */
    bool operator<(const MultiResVoxelKey& other) const {
        if (resolution_index != other.resolution_index) 
            return resolution_index < other.resolution_index;
        if (x != other.x) return x < other.x;
        if (y != other.y) return y < other.y;
        return z < other.z;
    }
    
    /**
     * @brief 等于运算符（纯整数比较，零精度误差）
     * 
     * @param other 待比较的另一个体素键
     * @return true 如果两个键表示同一个体素
     */
    bool operator==(const MultiResVoxelKey& other) const {
        return x == other.x && y == other.y && z == other.z && 
               resolution_index == other.resolution_index;
    }
};


struct MultiResVoxelData {
    Eigen::Vector3d point;             
    int intensity;                      

    MultiResVoxelData() : intensity(0) {}
    
    MultiResVoxelData(const Eigen::Vector3d& pt, int intens = 0) 
        : point(pt), intensity(intens) {}
};

// ==================== std::hash 特化（用于 unordered_map）====================
// ⚠️ 重要：std::unordered_map 对自定义类型**必须**提供哈希函数！
// 标准库只为基本类型（int, string等）提供默认哈希，自定义struct必须手动特化
// 如果注释掉此哈希函数，编译器会报错：
// error: no matching function for call to 'std::hash<MultiResVoxelKey>::operator()'
namespace std {
    template <>
    struct hash<MultiResVoxelKey> {
        std::size_t operator()(const MultiResVoxelKey& key) const {
            // 纯整数键 hash（无浮点精度问题）
            // 使用 Boost hash_combine 算法（工业级标准，低冲突率）
            size_t seed = 0;
            
            // Boost hash_combine: 黄金比例常数 + 位移混合
            auto hash_combine = [](size_t& seed, size_t value) {
                seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            };
            
            hash_combine(seed, static_cast<size_t>(key.x));
            hash_combine(seed, static_cast<size_t>(key.y));
            hash_combine(seed, static_cast<size_t>(key.z));
            hash_combine(seed, static_cast<size_t>(key.resolution_index));
            
            return seed;
        }
    };
}


// ==================== typedef 定义（必须在 hash 特化之后）====================
// 使用 std::unordered_map 提供 O(1) 查询性能（需要 hash 特化）
typedef std::unordered_map<MultiResVoxelKey, MultiResVoxelData> gt_voxelHashMap;

/**
 * @brief 多帧射线投射查询结果结构体
 * 
 * 封装了多帧融合射线投射查询的完整结果信息，
 * 包括命中状态、几何信息、质量评估等。
 */
struct RaycastResult {
    // 基本查询结果
    bool hit;                           // 是否成功命中体素并找到真值点
    Eigen::Vector3d point;              // 命中的真值点坐标（世界坐标系）
    
    // 查询配置和质量信息  
    double used_resolution;             // 实际使用的体素分辨率（米）
    double distance;                    // 从相机中心到命中点的距离（米）
    double reprojection_error;          // 在查询帧中的重投影误差（像素）
    
    // 调试和分析信息
    std::string failure_reason;         // 查询失败原因（用于调试）
    
    /**
     * @brief 默认构造函数，初始化为未命中状态
     */
    RaycastResult() 
        : hit(false), used_resolution(0.0f), distance(0.0),
          reprojection_error(std::numeric_limits<double>::max()),
          failure_reason("not_queried") {}
    
    /**
     * @brief 成功命中的构造函数
     * @param pt 命中的3D点
     * @param res 使用的分辨率
     */
    RaycastResult(const Eigen::Vector3d& pt, double res)
        : hit(true), point(pt), used_resolution(res), distance(0.0),
          reprojection_error(0.0), failure_reason("") {
        distance = pt.norm();
    }
};

/**
 * @brief 多分辨率体素地图类
 * 
 * 核心功能：
 * 1. 存储和管理多种分辨率（0.1m-1.0m）的真值点云体素地图
 * 2. 支持基于几何投影的自适应分辨率选择
 * 3. 提供高效的3D DDA射线投射查询
 * 4. 支持多帧射线融合以提高定位精度
 * 
 * 设计原则：
 * - 每个体素只存储一个代表点，避免存储冗余
 * - 自适应分辨率选择，根据观测几何确定查询精度
 * - 多帧融合策略，利用历史已优化位姿提高精度
 * - 内存友好的LRU缓存管理
 */
class MultiResolutionVoxelMap {
private:
    // 核心存储结构（使用 std::unordered_map 哈希表）
    gt_voxelHashMap multi_res_map;    // 多分辨率体素存储容器（使用哈希表，O(1)查询，需要自定义hash函数）
    std::vector<double> resolutions;                                // 支持的分辨率列表（0.1, 0.2, 0.3, ..., 1.0）
    
    // 线程安全保护（std::unordered_map 不是线程安全的！）
    mutable std::mutex map_mutex;                                   // 保护 multi_res_map 的互斥锁
    
    // 查询和搜索参数
    double max_search_distance;                                     // 射线搜索的最大距离（米）
    double step_size;                                               // 3D DDA算法的步长系数
    double min_confidence_threshold;                                // 接受查询结果的最小置信度阈值
    int max_fusion_frames;                                          // 多帧融合的最大帧数限制
    double query_default_focal_length;                              // raycastQuery2默认焦距（当无法从相机内参读取时使用）
    double query_default_optimal_resolution;                        // performRayIndexing默认分辨率（当未命中时用于动态阈值计算）
    double dynamic_threshold_offset;                                // raycastQuery2动态阈值常数偏移（原公式中的 +1.5）
    double dynamic_threshold_upper_bound_pixel;                     // performRayIndexing动态阈值超过上限时回退像素值
    double raycast_original_search_range;                           // raycastQuery2初始搜索范围（原始 original_search_range）
    
    // 性能优化相关
    bool enable_lru_cache;                                          // 是否启用LRU缓存管理
    size_t max_cache_size;                                          // 最大缓存体素数量
    double cache_cleanup_interval;                                  // 缓存清理间隔（秒）
    
    // 统计和调试信息
    mutable size_t query_count;                                     // 查询次数统计
    mutable size_t hit_count;                                       // 命中次数统计
    mutable double total_query_time;                                // 总查询时间统计
    mutable size_t collect_candidate_voxels_count;                  // collectCandidateVoxels 调用次数统计
    mutable size_t collect_candidate_voxels_empty_count;            // collectCandidateVoxels 返回空列表的次数统计
    
    // 友元类声明，允许voxelStereoVio访问私有成员
    friend class voxelStereoVio;
    
public:
    /**
     * @brief 构造函数，初始化默认参数
     */
    MultiResolutionVoxelMap() 
        : max_search_distance(120.0), step_size(0.1), min_confidence_threshold(0.6),
                                            max_fusion_frames(5), query_default_focal_length(458.0), query_default_optimal_resolution(0.01), dynamic_threshold_offset(1.5),
                                            dynamic_threshold_upper_bound_pixel(3.0),
                      raycast_original_search_range(0.75),
                    enable_lru_cache(true), max_cache_size(1000000),
          cache_cleanup_interval(60.0), query_count(0), hit_count(0), total_query_time(0.0),
          collect_candidate_voxels_count(0), collect_candidate_voxels_empty_count(0) {
        // 初始化默认分辨率列表：0.1m到0.3m
        resolutions = {0.1, 0.2, 0.3};
    }
    
    /**
     * @brief 析构函数，清理资源并输出统计信息
     */
    ~MultiResolutionVoxelMap() {
        // 输出统计信息（可选）
        if (query_count > 0) {
            double hit_rate = static_cast<double>(hit_count) / query_count;
            double avg_time = total_query_time / query_count;
            double empty_rate = (collect_candidate_voxels_count > 0) ? 
                static_cast<double>(collect_candidate_voxels_empty_count) / collect_candidate_voxels_count : 0.0;
        }
    }
    
    // // ==================== 地图构建接口 ====================
    
    
    // /**
    //  * @brief 增量式添加点云数据
    //  * @param new_points 新的点云数据
    //  * @param timestamp 时间戳
    //  */
    // void addPointCloudIncremental(const pcl::PointCloud<pcl::PointXYZ>::Ptr& new_points,
    //                              double timestamp);
    
    // ==================== 查询接口 ====================
    
    /**
     * @brief 单帧射线投射查询
     * 
     * 根据相机位姿和归一化像素坐标，通过3D DDA算法进行射线投射，
     * 自动选择合适的分辨率并返回命中的真值点。
     * 
     * @param camera_center 相机中心在世界坐标系中的位置
     * @param camera_rotation 相机旋转矩阵（世界坐标系到相机坐标系）
     * @param normalized_pixel 归一化像素坐标 [x/z, y/z]
     * @param pixel_coord 原始像素坐标 [u, v]
     * @param camera_K 相机内参矩阵（用于计算投影尺寸）
     * @return 查询结果，包含命中状态和点坐标
     */
    RaycastResult raycastQuery2(const Eigen::Vector3d& camera_center,
                              const Eigen::Matrix3d& camera_rotation,
                              const Eigen::Vector2d& normalized_pixel,
                              const Eigen::Vector2d& pixel_coord,
                              size_t camera_id,
                              const std::unordered_map<size_t, std::shared_ptr<cameraBase>>& cam_intrinsics_cameras,
                              double optimal_resolution = -1.0,
                              double distance_to_camera = -1.0,
                              double triangulated_depth = -1.0) const;                     
          
    // ==================== 状态查询和调试接口 ====================
    
    /**
     * @brief 获取各分辨率的体素数量统计
     * @return 分辨率到体素数量的映射
     */
    std::map<double, size_t> getVoxelCountByResolution() const;
    
    /**
     * @brief 获取总体素数量
     * @return 所有分辨率的体素总数
     */
    size_t getTotalVoxelCount() const;
    
    // /**
    //  * @brief 获取查询性能统计
    //  * @return 包含命中率、平均查询时间等信息的字符串
    //  */
    // std::string getPerformanceStats() const;
    
    /**
     * @brief 清空所有地图数据
     */
    void clearMap();

    // ==================== 查询参数只读接口 ====================
    double getQueryDefaultFocalLength() const { return query_default_focal_length; }
    double getQueryDefaultOptimalResolution() const { return query_default_optimal_resolution; }
    double getDynamicThresholdOffset() const { return dynamic_threshold_offset; }
    double getDynamicThresholdUpperBoundPixel() const { return dynamic_threshold_upper_bound_pixel; }
    double getRaycastOriginalSearchRange() const { return raycast_original_search_range; }
    
    /**
     * @brief 输出统计信息
     */
    void printStatistics() const {
        if (query_count > 0) {
            double hit_rate = static_cast<double>(hit_count) / query_count;
            double avg_time = total_query_time / query_count;
            std::cout << "[MultiResVoxelMap] Query Stats - Total: " << query_count 
                      << ", Hits: " << hit_count
                      << ", Hit rate: " << std::fixed << std::setprecision(2) << (hit_rate * 100) << "%"
                      << ", Avg time: " << std::fixed << std::setprecision(3) << avg_time << " ms" << std::endl;
        }
        
        if (collect_candidate_voxels_count > 0) {
            double empty_rate = static_cast<double>(collect_candidate_voxels_empty_count) / collect_candidate_voxels_count;
            std::cout << "[MultiResVoxelMap] CollectCandidateVoxels Stats - Total calls: " << collect_candidate_voxels_count
                      << ", Empty results: " << collect_candidate_voxels_empty_count
                      << ", Empty rate: " << std::fixed << std::setprecision(2) << (empty_rate * 100) << "%" << std::endl;
        }
    }
    
    
    /**
     * @brief 根据投影几何选择最佳查询分辨率
     * 
     * 核心算法：利用相机几何和相似三角形原理，
     * 计算像素在3D空间的投影尺寸，选择匹配的体素分辨率。
     * 
     * @param projected_footprint 像素在3D空间的投影尺寸
     * @return 最适合的体素分辨率
     */
    double selectOptimalResolution(double projected_footprint) const;
    
    
private:
    // ==================== 内部辅助函数 ====================
    
    /**
     * @brief 根据3D点坐标和分辨率计算体素键值
     * @param point 3D点坐标    
     * @param resolution 体素分辨率
     * @return 对应的体素键
     */
    MultiResVoxelKey getVoxelKey(const Eigen::Vector3d& point, double resolution) const;
    
    // /**
    //  * @brief 计算像素在指定3D位置的投影尺寸
    //  * 
    //  * 使用相似三角形原理：pixel_size / focal_length = footprint_size / distance
    //  * 
    //  * @param camera_center 相机中心位置
    //  * @param normalized_pixel 归一化像素坐标
    //  * @param intersection_point 射线与场景的交点
    //  * @param K 相机内参矩阵
    //  * @return 投影尺寸（米）
    //  */
    // double calculateProjectedFootprint(const Eigen::Vector3d& camera_center,
    //                                  const Eigen::Vector2d& normalized_pixel,
    //                                  const Eigen::Vector3d& intersection_point,
    //                                  const Eigen::Matrix3d& K) const;
    
    /**
     * @brief 3D DDA射线投射算法核心实现
     * @param ray_origin 射线起点
     * @param ray_direction 射线方向（单位向量）
     * @param resolution 使用的体素分辨率
     * @param max_distance 最大搜索距离
     * @param estimated_depth 估计深度（可选），用于优化搜索起点，默认-1.0表示从射线原点开始
     * @param search_range 搜索范围（可选），在估计深度附近的搜索范围，默认5.0米
     * @return 首个命中的体素数据（值拷贝），若未命中则返回 std::nullopt
     */
    std::optional<MultiResVoxelData> performDDATraversal(const Eigen::Vector3d& ray_origin,
                                                         const Eigen::Vector3d& ray_direction,
                                                         double resolution,
                                                         double distance_to_camera = -1.0,
                                                         double search_range = 15.0) const;
    
    /**
     * @brief 拟合平面圆（面圆）
     * @param points 输入点云
     * @param circle_center 输出：圆心
     * @param circle_radius 输出：半径
     * @param plane_normal 输出：平面法向量
     * @param plane_centroid 输出：平面质心（PCA拟合的参考点）
     * @return 是否拟合成功
     */
    bool fitPlanarCircle(const std::vector<Eigen::Vector3d>& points,
                        Eigen::Vector3d& circle_center,
                        double& circle_radius,
                        Eigen::Vector3d& plane_normal,
                        Eigen::Vector3d& plane_centroid) const;
    
    /**
     * @brief 计算射线与平面圆的交点
     * @param ray_origin 射线起点
     * @param ray_direction 射线方向（单位向量）
     * @param circle_center 圆心
     * @param circle_radius 半径
     * @param plane_normal 平面法向量
     * @param plane_centroid 平面质心（用于定义平面）
     * @param intersection 输出：交点
     * @return 是否存在交点
     */
    bool rayCircleIntersection(const Eigen::Vector3d& ray_origin,
                              const Eigen::Vector3d& ray_direction,
                              const Eigen::Vector3d& circle_center,
                              double circle_radius,
                              const Eigen::Vector3d& plane_normal_input,
                              const Eigen::Vector3d& plane_centroid,
                              Eigen::Vector3d& intersection) const;
};


class voxelStereoVio
{
private:

    void processImu(imuData &imu_data);

    void processImage(cameraData &image_measurements_const);

    void featureUpdate_gt_msckf(cameraData &image_measurements);

    void triangulateActiveTracks(cameraData &image_measurements);

    void getRecentVoxel(double timestamp, pcl::PointCloud<pcl::PointXYZI>::Ptr voxels_visit);

    bool tryToInitialize(cameraData &image_measurements);

    void loadGlobalPoses(const std::string &filename);

    GTPose interpolateGTPose(double query_timestamp);

    void Change_UTM_Coordinate(double &timestamp, Eigen::MatrixXd &covariance, std::vector<std::shared_ptr<baseType>> &order, 
    std::shared_ptr<imuState> imu_state_);


    // ==================== 多分辨率真值地图查询接口 ====================
    
    /**
     * @brief 初始化真值地图系统
     * 
     * 从指定路径加载真值点云数据，构建多分辨率体素地图。
     * 支持的格式：PCD, PLY, TXT等常见点云格式。
     * 
     * @return true 如果初始化成功
     */
    bool initializeGroundTruthMap();
    
    /**
     * @brief 辅助函数：计算体素键值
     * @param point 3D点坐标
     * @param resolution 体素分辨率
     * @return 对应的体素键
     */
    MultiResVoxelKey getVoxelKey(const Eigen::Vector3d& point, double resolution) const;

    std::queue<cameraData> camera_buffer;

    std::queue<imuData> imu_buffer;

    std::map<int, double> camera_last_timestamp;

    odometryOptions odometry_options;

    std::shared_ptr<trackKLT> featureTracker;

    std::shared_ptr<inertialInitializer> initializer_ptr;

    double startup_time;

    double current_time;

    double time_newest_imu;

    std::vector<std::pair<double, std::pair<Eigen::Vector3d, Eigen::Vector3d>>> imu_meas;
    std::vector<imuState> imu_states;

    std::shared_ptr<state> state_ptr;

    imuData last_imu_data;

    std::shared_ptr<propagator> propagator_ptr;

    std::shared_ptr<updaterMsckf> updaterMsckf_ptr;

    std::shared_ptr<updaterSlam> updaterSlam_ptr;

    std::shared_ptr<gammaPixel> gammaPixel_ptr;

    std::shared_ptr<frame> newest_fh;

    std::vector<double> camera_queue_init;

    std::vector<std::shared_ptr<frame>> frame_queue_init;

    voxelHashMap voxel_map;
    voxelHashMap gt_voxel_map;  // 真值地图对应的体素地图

    // ==================== 多分辨率真值体素地图存储 ====================
    
    MultiResolutionVoxelMap multi_res_gt_map;          // 多分辨率真值体素地图实例
    
    // 配置参数
    bool use_ground_truth_map;                          // 是否启用真值地图功能的全局开关
    bool use_new;                                       // 控制是否使用active_tracks_pos_world_new
    bool enable_global_map_marginalization_check;      // 是否启用全局map_points的额外边缘化检查
    std::string ground_truth_map_path;                  // 真值点云地图文件路径
    std::string ground_truth_pose_path;                 // 全局位姿文件路径（interpolateGTPose使用）
    std::vector<double> ground_truth_map_resolutions;   // 真值地图分辨率列表（从配置文件读取）


    int frame_count;

    MarginalizeStatus marginalize_status;

    double last_time_image;

    std::vector<Eigen::Vector3d> good_features_msckf;

    double active_tracks_time = -1;
    std::unordered_map<size_t, Eigen::Vector3d> active_tracks_pos_world; // active_tracks_posinG
    std::unordered_map<size_t, Eigen::Vector3d> active_tracks_pos_world_new; // active_tracks_posinG

    std::unordered_map<size_t, Eigen::Vector3d> active_tracks_uvd;
    cv::Mat active_image;
    std::map<size_t, Eigen::Matrix3d> active_feat_linsys_A;
    std::map<size_t, Eigen::Vector3d> active_feat_linsys_b;
    std::map<size_t, int> active_feat_linsys_count;

    std::vector<Eigen::Matrix<short, 3, 1>> recent_voxels;

    std::ofstream of_statistics;
    boost::posix_time::ptime rT1, rT2, rT3, rT4, rT5, rT6, rT7;

    double timelastupdate = -1;
    double distance = 0;

    // time test
    double sum_time_1 = 0.0;
    double sum_time_2 = 0.0;
    double sum_time_3 = 0.0;
    double sum_time_4 = 0.0;
    double sum_time_5 = 0.0;
    double sum_time_6 = 0.0;
    double sum_time_7 = 0.0;

    double sum_time_10 = 0.0;

    double sum_raycast_time = 0.0;


    double sum_time_sum = 0.0;

    int n_time_1 = 0;
    int n_time_2 = 0;
    int n_time_3 = 0;
    int n_time_4 = 0;
    int n_time_5 = 0;
    int n_time_6 = 0;
    int n_time_7 = 0;

    int n_time_10 = 0;

    int n_raycast_time = 0;

    int n_time_sum = 0;
    // time test

public:

    voxelStereoVio();

    void readParameters();

    void allocateMemory();

    void initialValue();

    void imuHandler(const sensor_msgs::Imu::ConstPtr &msg);

    void stereoImageHandler(const sensor_msgs::ImageConstPtr &msg_0, const sensor_msgs::ImageConstPtr &msg_1, int cam_0, int cam_1);

    void run();

    ros::NodeHandle nh;

    std::string image_left_topic;
    std::string image_right_topic;
    std::string imu_topic;

    ros::Subscriber sub_imu;
    std::vector<ros::Subscriber> subs_cam;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> SyncStereoImage;
    std::vector<std::shared_ptr<message_filters::Synchronizer<SyncStereoImage>>> sync_cam;
    std::vector<std::shared_ptr<message_filters::Subscriber<sensor_msgs::Image>>> sync_subs_cam;

    // display
    void pubFeatImage(cv::Mat &stereo_image, double &timestamp);
    void pubOdometry(std::shared_ptr<state> state_ptr, double &timestamp);
    void setPoseStamp(geometry_msgs::PoseStamped &body_pose_out, std::shared_ptr<state> state_ptr);
    void pubPath(std::shared_ptr<state> state_ptr, double &timestamp);
    void pubHistoryPoints(pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_history, double &timestamp);
    void pubWindowPoints(std::shared_ptr<state> state_ptr, double &timestamp);
    void pubHistoryVoxels(pcl::PointCloud<pcl::PointXYZI>::Ptr voxels_history, double &timestamp);
    void pubVisitVoxels(pcl::PointCloud<pcl::PointXYZI>::Ptr voxels_visit, double &timestamp);

    image_transport::ImageTransport it;
    image_transport::Publisher pub_feat_image;

    ros::Publisher pub_odom;
    ros::Publisher pub_path;
    ros::Publisher pub_points_history;
    ros::Publisher pub_points_window;
    ros::Publisher pub_voxels_history;
    ros::Publisher pub_voxels_visit;

    geometry_msgs::PoseStamped msg_body_pose;
    nav_msgs::Odometry odom;
    nav_msgs::Path path;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_history;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr points_window;
    pcl::PointCloud<pcl::PointXYZI>::Ptr voxels_history;
    pcl::PointCloud<pcl::PointXYZI>::Ptr voxels_visit;
    // display

    Eigen::MatrixXd covariance_;
    std::vector<std::shared_ptr<baseType>> order_;

    GTPose gt_pose;
    
    // 坐标系变换矩阵（从旧世界系到新世界系）
    Eigen::Matrix3d R_GoldToGnew_;
    bool has_coordinate_transform_;  // 标记是否已执行坐标系变换
    

};


