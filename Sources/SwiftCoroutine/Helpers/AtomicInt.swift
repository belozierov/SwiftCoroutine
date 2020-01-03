//
//  AtomicInt.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.12.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif
import Foundation

struct AtomicInt {
    
    private var _value: Int
    
    init(value: Int) {
        _value = value
    }
    
    @inlinable var value: Int {
        get {
            atomic_thread_fence(memory_order_seq_cst)
            return _value
        }
        set {
            let value = OpaquePointer(UnsafeMutablePointer(&_value))
            while true {
                var oldValue = self.value
                if __compare(value, &oldValue, newValue) != 0 { return }
            }
        }
    }
    
}

@propertyWrapper
public struct AtomicIntRepresentable<T: RawRepresentable> where T.RawValue == Int {
    
    private var atomic: AtomicInt
    
    public init(wrappedValue value: T) {
        atomic = AtomicInt(value: value.rawValue)
    }
    
    public var wrappedValue: T {
        get { T(rawValue: atomic.value)! }
        set { atomic.value = newValue.rawValue }
    }
    
}


