//
//  AtomicEnum.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 24.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if SWIFT_PACKAGE
import CCoroutine
#endif

internal struct AtomicEnum<T: RawRepresentable> where T.RawValue == Int {
    
    private var _value: Int
    
    init(value: T) {
        _value = value.rawValue
    }
    
    var value: T {
        get { T(rawValue: _value)! }
        set {
            withUnsafeMutablePointer(to: &_value) {
                __atomicStore(OpaquePointer($0), newValue.rawValue)
            }
        }
    }
    
    mutating func update(_ newValue: T) -> T {
        withUnsafeMutablePointer(to: &_value) {
            T(rawValue: __atomicExchange(OpaquePointer($0), newValue.rawValue))!
        }
    }
    
}
