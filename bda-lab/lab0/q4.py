def binary_search(arr, target):
    low, high = 0, len(arr) - 1

    while low <= high:
        mid = (low + high) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            low = mid + 1
        else:
            high = mid - 1

    return -1
 
index = binary_search([1, 3, 5, 7, 9, 11, 13, 15, 17, 19], 7)
print(f"Element found at {index}." if index != -1 else f"Element not found.")