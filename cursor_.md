# 안쓰는 함수 찾기
_Exported on 8/28/2025 at 11:14:00 GMT+9 from Cursor (1.0.0)_

---

**User**

 !!! 안쓰는 함수가 있어?

---

**Cursor**

코드를 분석해서 사용되지 않는 함수들을 찾아보겠습니다.

이제 현재 파일에서 정의된 함수들과 실제로 호출되는 함수들을 비교해보겠습니다.

이제 각 함수가 실제로 호출되는지 확인해보겠습니다.

분석 결과를 정리하면, 다음 함수들이 **사용되지 않는 함수들**입니다:

## 🗑️ 사용되지 않는 함수들

1. **`draw_enhanced_debug_image`** (라인 1193-1219)
   - 정의되어 있지만 어디서도 호출되지 않음
   - TF reprojection을 그리는 함수이지만 실제로는 `draw_keypoint_analysis_on_debug_image`와 `draw_tf_reprojection_only`를 사용

2. **중복된 `shorten_axis` 함수들**:
   - 라인 1063: `draw_keypoint_analysis_on_debug_image` 내부의 중첩 함수
   - 라인 1532: `draw_tf_reprojection_only` 내부의 중첩 함수
   - 동일한 기능을 하는 함수가 두 곳에서 중복 정의됨

## 🧹 정리 제안

사용되지 않는 함수들을 제거해서 코드를 깔끔하게 정리하시겠습니까?

1. `draw_enhanced_debug_image` 함수 제거 (라인 1193-1219)
2. 중복된 `shorten_axis` 함수들을 하나의 클래스 메소드로 통합

이렇게 정리하면 코드가 더 깔끔해지고 유지보수가 쉬워집니다.

---

**User**

정!! 정리해줘

---

**Cursor**

네! 사용되지 않는 함수들을 정리해드리겠습니다.

```python
  def reproject_tf_to_image(self, tf_info, camera_matrix):
    """
    Reproject TF coordinate system back to image for visualization
    """
    reprojected_points = {}
    
    for fruit_id, tf_data in tf_info.items():
      position = tf_data['position']  # 3D position
      rotation_matrix = tf_data['rotation_matrix']  # 3x3 rotation matrix
      
      # Define coordinate system axes in 3D (더 짧은 길이)
      axis_length = 0.02  # 5cm에서 2cm로 줄임
      origin_3d = np.array(position)
      x_axis_3d = origin_3d + rotation_matrix[:, 0] * axis_length  # X axis (red)
      y_axis_3d = origin_3d + rotation_matrix[:, 1] * axis_length  # Y axis (green)  
      z_axis_3d = origin_3d + rotation_matrix[:, 2] * axis_length  # Z axis (blue)
      
      # Project to image coordinates
      def project_3d_to_image(point_3d):
        x, y, z = point_3d
        if z <= 0:  # 유효하지 않은 깊이 체크
          return None
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return (int(u), int(v))
      
      if position[2] > 0:  # Valid depth
        origin_2d = project_3d_to_image(origin_3d)
        x_axis_2d = project_3d_to_image(x_axis_3d)
        y_axis_2d = project_3d_to_image(y_axis_3d)
        z_axis_2d = project_3d_to_image(z_axis_3d)
        
        # 모든 투영점이 유효한지 확인
        if all(pt is not None for pt in [origin_2d, x_axis_2d, y_axis_2d, z_axis_2d]):
          reprojected_points[fruit_id] = {
            'origin': origin_2d,
            'x_axis': x_axis_2d,
            'y_axis': y_axis_2d,
            'z_axis': z_axis_2d
          }
    
    return reprojected_points

  def shorten_axis(self, origin, axis_end, max_length=25):
    """Helper function to shorten axis visualization lines"""
    dx = axis_end[0] - origin[0]
    dy = axis_end[1] - origin[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length > max_length:
      scale = max_length / length
      new_x = int(origin[0] + dx * scale)
      new_y = int(origin[1] + dy * scale)
      return (new_x, new_y)
    return axis_end

  def bbox_callback(self, msg):
```

```python
  def reproject_tf_to_image(self, tf_info, camera_matrix):
    """
    Reproject TF coordinate system back to image for visualization
    """
    reprojected_points = {}
    
    for fruit_id, tf_data in tf_info.items():
      position = tf_data['position']  # 3D position
      rotation_matrix = tf_data['rotation_matrix']  # 3x3 rotation matrix
      
      # Define coordinate system axes in 3D (더 짧은 길이)
      axis_length = 0.02  # 5cm에서 2cm로 줄임
      origin_3d = np.array(position)
      x_axis_3d = origin_3d + rotation_matrix[:, 0] * axis_length  # X axis (red)
      y_axis_3d = origin_3d + rotation_matrix[:, 1] * axis_length  # Y axis (green)  
      z_axis_3d = origin_3d + rotation_matrix[:, 2] * axis_length  # Z axis (blue)
      
      # Project to image coordinates
      def project_3d_to_image(point_3d):
        x, y, z = point_3d
        if z <= 0:  # 유효하지 않은 깊이 체크
          return None
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return (int(u), int(v))
      
      if position[2] > 0:  # Valid depth
        origin_2d = project_3d_to_image(origin_3d)
        x_axis_2d = project_3d_to_image(x_axis_3d)
        y_axis_2d = project_3d_to_image(y_axis_3d)
        z_axis_2d = project_3d_to_image(z_axis_3d)
        
        # 모든 투영점이 유효한지 확인
        if all(pt is not None for pt in [origin_2d, x_axis_2d, y_axis_2d, z_axis_2d]):
          reprojected_points[fruit_id] = {
            'origin': origin_2d,
            'x_axis': x_axis_2d,
            'y_axis': y_axis_2d,
            'z_axis': z_axis_2d
          }
    
    return reprojected_points

  #def draw_enhanced_debug_image(self, debug_img, keypoint_analysis, tf_reprojections):
    """
    Draw enhanced debug image with TF reprojections
    """
    for fruit_id, analysis in keypoint_analysis.items():
      # ... existing keypoint drawing code ...
      
      # Draw TF reprojection if available
      if fruit_id in tf_reprojections:
        tf_proj = tf_reprojections[fruit_id]
        origin = tf_proj['origin']
        
        # Draw coordinate axes
        cv2.arrowedLine(debug_img, origin, tf_proj['x_axis'], (0, 0, 255), 3)    # X-axis: Red
        cv2.arrowedLine(debug_img, origin, tf_proj['y_axis'], (0, 255, 0), 3)   # Y-axis: Green  
        cv2.arrowedLine(debug_img, origin, tf_proj['z_axis'], (255, 0, 0), 3)   # Z-axis: Blue
        
        # Draw origin point
        cv2.circle(debug_img, origin, 8, (255, 255, 255), -1)
        cv2.circle(debug_img, origin, 10, (0, 0, 0), 2)
        
        # Add axis labels
        cv2.putText(debug_img, 'X', tf_proj['x_axis'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(debug_img, 'Y', tf_proj['y_axis'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(debug_img, 'Z', tf_proj['z_axis'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return debug_img

->

  def reproject_tf_to_image(self, tf_info, camera_matrix):
    """
    Reproject TF coordinate system back to image for visualization
    """
    reprojected_points = {}
    
    for fruit_id, tf_data in tf_info.items():
      position = tf_data['position']  # 3D position
      rotation_matrix = tf_data['rotation_matrix']  # 3x3 rotation matrix
      
      # Define coordinate system axes in 3D (더 짧은 길이)
      axis_length = 0.02  # 5cm에서 2cm로 줄임
      origin_3d = np.array(position)
      x_axis_3d = origin_3d + rotation_matrix[:, 0] * axis_length  # X axis (red)
      y_axis_3d = origin_3d + rotation_matrix[:, 1] * axis_length  # Y axis (green)  
      z_axis_3d = origin_3d + rotation_matrix[:, 2] * axis_length  # Z axis (blue)
      
      # Project to image coordinates
      def project_3d_to_image(point_3d):
        x, y, z = point_3d
        if z <= 0:  # 유효하지 않은 깊이 체크
          return None
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return (int(u), int(v))
      
      if position[2] > 0:  # Valid depth
        origin_2d = project_3d_to_image(origin_3d)
        x_axis_2d = project_3d_to_image(x_axis_3d)
        y_axis_2d = project_3d_to_image(y_axis_3d)
        z_axis_2d = project_3d_to_image(z_axis_3d)
        
        # 모든 투영점이 유효한지 확인
        if all(pt is not None for pt in [origin_2d, x_axis_2d, y_axis_2d, z_axis_2d]):
          reprojected_points[fruit_id] = {
            'origin': origin_2d,
            'x_axis': x_axis_2d,
            'y_axis': y_axis_2d,
            'z_axis': z_axis_2d
          }
    
    return reprojected_points

  def shorten_axis(self, origin, axis_end, max_length=25):
    """Helper function to shorten axis visualization lines"""
    dx = axis_end[0] - origin[0]
    dy = axis_end[1] - origin[1]
    length = math.sqrt(dx*dx + dy*dy)
    if length > max_length:
      scale = max_length / length
      new_x = int(origin[0] + dx * scale)
      new_y = int(origin[1] + dy * scale)
      return (new_x, new_y)
    return axis_end
```

이제 중복된 `shorten_axis` 함수들을 새로 만든 클래스 메소드로 교체하겠습니다.

```python
        # TF 축들을 더 짧게 만들기
        def shorten_axis(origin, axis_end, max_length=20):
          dx = axis_end[0] - origin[0]
          dy = axis_end[1] - origin[1]
          length = math.sqrt(dx*dx + dy*dy)
          if length > max_length:
            scale = max_length / length
            new_x = int(origin[0] + dx * scale)
            new_y = int(origin[1] + dy * scale)
            return (new_x, new_y)
          return axis_end
        
        # 축들을 더 짧게
        short_x_axis = shorten_axis(origin, tf_proj['x_axis'], 20)
        short_y_axis = shorten_axis(origin, tf_proj['y_axis'], 20)  
        short_z_axis = shorten_axis(origin, tf_proj['z_axis'], 20)

->

        # 축들을 더 짧게
        short_x_axis = self.shorten_axis(origin, tf_proj['x_axis'], 20)
        short_y_axis = self.shorten_axis(origin, tf_proj['y_axis'], 20)  
        short_z_axis = self.shorten_axis(origin, tf_proj['z_axis'], 20)
```

```python
        def shorten_axis(origin, axis_end, max_length=20):
          dx = axis_end[0] - origin[0]
          dy = axis_end[1] - origin[1]
          length = math.sqrt(dx*dx + dy*dy)
          if length > max_length:
            scale = max_length / length
            new_x = int(origin[0] + dx * scale)
            new_y = int(origin[1] + dy * scale)
            return (new_x, new_y)
          return axis_end
        
        # 축들을 더 짧게
        short_x_axis = shorten_axis(origin, tf_proj['x_axis'], 20)
        short_y_axis = shorten_axis(origin, tf_proj['y_axis'], 20)  
        short_z_axis = shorten_axis(origin, tf_proj['z_axis'], 20)

->

        # 축들을 더 짧게
        short_x_axis = self.shorten_axis(origin, tf_proj['x_axis'], 20)
        short_y_axis = self.shorten_axis(origin, tf_proj['y_axis'], 20)  
        short_z_axis = self.shorten_axis(origin, tf_proj['z_axis'], 20)
```

이제 `draw_tf_reprojection_only` 함수에서도 중복된 `shorten_axis` 함수를 교체하겠습니다.

```python
        # TF 축들을 적당한 길이로
        def shorten_axis(origin, axis_end, max_length=25):
          dx = axis_end[0] - origin[0]
          dy = axis_end[1] - origin[1]
          length = math.sqrt(dx*dx + dy*dy)
          if length > max_length:
            scale = max_length / length
            new_x = int(origin[0] + dx * scale)
            new_y = int(origin[1] + dy * scale)
            return (new_x, new_y)
          return axis_end
        
        # 축들을 적당한 길이로
        short_z_axis = shorten_axis(origin, tf_proj['z_axis'], 25)

->

        # 축들을 적당한 길이로
        short_z_axis = self.shorten_axis(origin, tf_proj['z_axis'], 25)
```

완료되었습니다! 🎉

## ✅ 정리 완료

다음 사항들을 정리했습니다:

### 🗑️ 제거된 것들:
1. **`draw_enhanced_debug_image` 함수** (라인 1193-1219) - 사용되지 않던 함수 완전 제거
2. **중복된 `shorten_axis` 중첩 함수들** - 두 곳에서 중복 정의되던 함수들 제거

### ➕ 추가된 것들:
1. **`shorten_axis` 클래스 메소드** - 공통으로 사용할 수 있는 헬퍼 함수로 통합
2. 기존 중복 함수 호출을 `self.shorten_axis()` 메소드 호출로 변경

### 📈 개선 효과:
- **코드 중복 제거**: 동일한 기능의 함수가 여러 곳에 중복 정의되던 문제 해결
- **유지보수성 향상**: 하나의 함수만 수정하면 모든 곳에 적용
- **가독성 개선**: 불필요한 코드 제거로 더 깔끔한 구조
- **메모리 효율**: 사용되지 않는 함수 제거

이제 코드가 훨씬 깔끔하고 효율적이 되었습니다! 🚀

---

**User**

! 이건 그래서 어떻게 repreojection 되는 거야?

---

**Cursor**

이 코드에서 **TF reprojection**이 어떻게 동작하는지 설명해드릴게요! 🎯

## 📍 TF Reprojection 과정

### 1. **3D TF 생성** (라인 818-929)
```python
def create_fruit_oriented_tf_from_keypoints():
    # 꼭지(yellow) 키포인트를 3D 원점으로 사용
    x = (yellow_point[0] - self.cx) * depth_m / self.fx
    y = (yellow_point[1] - self.cy) * depth_m / self.fy  
    z = depth_m
    
    # Z축을 끝→꼭지 방향으로 회전시킨 3D 좌표계 생성
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # TF 정보 저장
    self.latest_tf_info[fruit_id] = {
        'position': (x, y, z),           # 3D 위치
        'rotation_matrix': rotation_matrix,  # 3x3 회전 행렬
        'z_axis_angle': smoothed_angle   # Z축 각도
    }
```

### 2. **3D → 2D Reprojection** (라인 1139-1180)
```python
def reproject_tf_to_image():
    # 3D 좌표계 축들 정의 (2cm 길이)
    origin_3d = np.array(position)
    x_axis_3d = origin_3d + rotation_matrix[:, 0] * 0.02  # X축 (빨강)
    y_axis_3d = origin_3d + rotation_matrix[:, 1] * 0.02  # Y축 (초록)  
    z_axis_3d = origin_3d + rotation_matrix[:, 2] * 0.02  # Z축 (파랑)
    
    # 3D → 2D 투영 (카메라 내부 파라미터 사용)
    def project_3d_to_image(point_3d):
        x, y, z = point_3d
        u = self.fx * x / z + self.cx  # 픽셀 u 좌표
        v = self.fy * y / z + self.cy  # 픽셀 v 좌표
        return (int(u), int(v))
```

### 3. **이미지에 그리기** (라인 1458-1531)

#### **Detailed 이미지** (모든 정보 포함):
- 🔴 빨간 화살표: X축
- 🟢 초록 화살표: Y축  
- 🔵 파란 화살표: Z축
- ⚪ 흰 원: TF 원점 (꼭지 위치)

#### **TF-only 이미지** (깔끔한 버전):
- ⚪ 흰 화살표: Z축만 표시 (과일 방향)
- ⚪ 흰 원: TF 원점
- 📝 라벨: "TF_1", "TF_2" 등

## 🔄 전체 플로우

```mermaid
graph TD
    A[키포인트 검출] --> B[3D TF 생성]
    B --> C[TF 정보 저장]
    C --> D[3D→2D 투영]
    D --> E[이미지에 그리기]
    
    B1[꼭지 = 원점] --> B
    B2[끝→꼭지 = Z축 방향] --> B
    B3[뎁스 = Z 좌표] --> B
    
    D1[카메라 내부파라미터] --> D
    D2[fx, fy, cx, cy] --> D
    
    E1[Detailed 이미지] --> E
    E2[TF-only 이미지] --> E
```

## 🎨 시각화 결과

실제로 이미지에서 보이는 것:

1. **과일 키포인트**: 끝(흰색), 꼭지(노란색), 중간(빨간색)
2. **TF 좌표계**: 꼭지에서 시작하는 3D 축들이 2D로 투영됨
3. **Z축 방향**: 과일이 향하는 방향을 나타냄
4. **위치 정보**: 깊이(mm) 표시

이렇게 해서 **3D 공간의 TF가 2D 이미지에 정확히 reprojection**되어 시각화됩니다! 🚀

궁금한 부분이 더 있으면 언제든 물어보세요! 😊

```python
def create_fruit_oriented_tf_from_keypoints():
    # 꼭지(yellow) 키포인트를 3D 원점으로 사용
    x = (yellow_point[0] - self.cx) * depth_m / self.fx
    y = (yellow_point[1] - self.cy) * depth_m / self.fy  
    z = depth_m
    
    # Z축을 끝→꼭지 방향으로 회전시킨 3D 좌표계 생성
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # TF 정보 저장
    self.latest_tf_info[fruit_id] = {
        'position': (x, y, z),           # 3D 위치
        'rotation_matrix': rotation_matrix,  # 3x3 회전 행렬
        'z_axis_angle': smoothed_angle   # Z축 각도
    }
```

```python
def reproject_tf_to_image():
    # 3D 좌표계 축들 정의 (2cm 길이)
    origin_3d = np.array(position)
    x_axis_3d = origin_3d + rotation_matrix[:, 0] * 0.02  # X축 (빨강)
    y_axis_3d = origin_3d + rotation_matrix[:, 1] * 0.02  # Y축 (초록)  
    z_axis_3d = origin_3d + rotation_matrix[:, 2] * 0.02  # Z축 (파랑)
    
    # 3D → 2D 투영 (카메라 내부 파라미터 사용)
    def project_3d_to_image(point_3d):
        x, y, z = point_3d
        u = self.fx * x / z + self.cx  # 픽셀 u 좌표
        v = self.fy * y / z + self.cy  # 픽셀 v 좌표
        return (int(u), int(v))
```

```plaintext
graph TD
    A[키포인트 검출] --> B[3D TF 생성]
    B --> C[TF 정보 저장]
    C --> D[3D→2D 투영]
    D --> E[이미지에 그리기]
    
    B1[꼭지 = 원점] --> B
    B2[끝→꼭지 = Z축 방향] --> B
    B3[뎁스 = Z 좌표] --> B
    
    D1[카메라 내부파라미터] --> D
    D2[fx, fy, cx, cy] --> D
    
    E1[Detailed 이미지] --> E
    E2[TF-only 이미지] --> E
```

