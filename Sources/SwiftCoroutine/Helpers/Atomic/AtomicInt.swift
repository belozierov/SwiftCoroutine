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

@propertyWrapper
public struct AtomicInt {
    
    private var value: Int
    
    public init(wrappedValue value: Int) {
        self.value = value
    }
    
     public var wrappedValue: Int {
        get { value }
        set {
            let pointer = OpaquePointer(UnsafeMutablePointer(&value))
            var oldValue: Int
            repeat { oldValue = value }
                while __compare(pointer, &oldValue, newValue) == 0
        }
    }
    
    public mutating func increase() -> Int {
        let pointer = OpaquePointer(UnsafeMutablePointer(&value))
        var oldValue = value, newValue: Int;
        repeat { newValue = oldValue + 1 }
            while __compare(pointer, &oldValue, newValue) == 0
        return newValue
    }
    
}
