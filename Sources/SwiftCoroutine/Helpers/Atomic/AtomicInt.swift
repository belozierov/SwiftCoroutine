//
//  AtomicInt.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 24.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif

@propertyWrapper @usableFromInline
struct AtomicInt {
    
    private var value: Int
    
    init(wrappedValue value: Int) {
        self.value = value
    }
    
    var wrappedValue: Int {
        get { value }
        set {
            let pointer = OpaquePointer(UnsafeMutablePointer(&value))
            var oldValue: Int
            repeat { oldValue = value }
                while __compare(pointer, &oldValue, newValue) == 0
        }
    }
    
    @inlinable var projectedValue: AtomicInt {
        get { self }
        set { self = newValue }
    }
    
    @discardableResult mutating func increase() -> Int {
        let pointer = OpaquePointer(UnsafeMutablePointer(&value))
        var oldValue = value, newValue: Int
        repeat { newValue = oldValue + 1 }
            while __compare(pointer, &oldValue, newValue) == 0
        return newValue
    }
    
    mutating func update(from: Int, to: Int) -> Bool {
        let pointer = OpaquePointer(UnsafeMutablePointer(&value))
        var oldValue = value
        repeat {
            if oldValue != from { return false }
        } while __compare(pointer, &oldValue, to) == 0
        return true
    }
    
}
