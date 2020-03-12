//
//  CoFutureTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 31.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFutureTests: XCTestCase {
    
    func testResult1() {
        let promise = CoPromise<Bool>()
        XCTAssertNil(promise.result)
        promise.success(true)
        promise.success(false)
        promise.cancel()
        XCTAssertEqual(promise.result, true)
    }
    
    func testResult2() {
        let future = CoFuture(result: .success(true))
        XCTAssertEqual(future.result, true)
        future.cancel()
        XCTAssertEqual(future.result, true)
    }
    
    
}
